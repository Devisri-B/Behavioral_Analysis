"""
Drift detection - monitors for distribution shifts using K-S test, PSI, and prediction shift.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from typing import Dict, List, Tuple, Optional, cast
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# CSV storage directory for evaluation data
EVAL_DIR = Path('output/evaluations')
EVAL_DIR.mkdir(parents=True, exist_ok=True)


class DriftDetector:
    """Detects model/data drift using statistical tests."""
    
    def __init__(
        self,
        ks_threshold: float = 0.05,
        psi_threshold: float = 0.15,
        pred_shift_threshold: float = 0.20,
        min_window_size: int = 100
    ):
        self.ks_threshold = ks_threshold
        self.psi_threshold = psi_threshold
        self.pred_shift_threshold = pred_shift_threshold
        self.min_window_size = min_window_size
        
        # State for tracking
        self.baseline_scores = None
        self.baseline_pred_counts = None
        self.window_history = []  # List of {timestamp, pred_counts, drift_detected}
    
    def fit_baseline(self, scores: np.ndarray, predictions: np.ndarray, num_classes: int = 3) -> None:
        """Fit baseline drift model on training data."""
        scores = np.asarray(scores).flatten()
        predictions = np.asarray(predictions, dtype=int).flatten()
        
        if len(scores) < self.min_window_size:
            logger.warning(
                f"Baseline sample size {len(scores)} < {self.min_window_size}. "
                "Drift detection may be unreliable."
            )
        
        self.baseline_scores = scores
        
        # Baseline prediction distribution
        unique, counts = np.unique(predictions, return_counts=True)
        self.baseline_pred_counts = {int(u): int(c) for u, c in zip(unique, counts)}
        
        logger.info(f"Drift detector baseline fitted. Baseline pred counts: {self.baseline_pred_counts}")
    
    def detect_drift(self, scores: np.ndarray, predictions: np.ndarray, window_id: Optional[str] = None) -> Dict:
        """Detect drift in new data vs baseline."""
        if self.baseline_scores is None:
            logger.warning("Baseline not fitted. Call fit_baseline() first.")
            return {'drift_detected': False, 'reason': 'no_baseline'}
        
        scores = np.asarray(scores).flatten()
        predictions = np.asarray(predictions, dtype=int).flatten()
        
        if len(scores) < self.min_window_size:
            return {'drift_detected': False, 'reason': 'insufficient_samples', 'sample_count': len(scores)}
        
        report = {'window_id': window_id or datetime.utcnow().isoformat(), 'sample_count': len(scores), 'timestamp': datetime.utcnow().isoformat(), 'methods': {}}
        
        # K-S test
        ks_result = ks_2samp(self.baseline_scores, scores)
        ks_stat, ks_pval = cast(float, ks_result[0]), cast(float, ks_result[1])
        report['methods']['ks_test'] = {'statistic': ks_stat, 'p_value': ks_pval, 'threshold': self.ks_threshold, 'drift_detected': ks_pval < self.ks_threshold}
        
        # PSI
        psi = self._compute_psi(self.baseline_scores, scores)
        report['methods']['psi'] = {'psi': float(psi), 'threshold': self.psi_threshold, 'drift_detected': psi > self.psi_threshold}
        
        # Prediction shift
        unique, counts = np.unique(predictions, return_counts=True)
        new_pred_counts = {int(u): int(c) for u, c in zip(unique, counts)}
        baseline_counts = self.baseline_pred_counts or {}
        pred_shift, shift_details = self._compute_prediction_shift(baseline_counts, new_pred_counts)
        report['methods']['prediction_shift'] = {'shift': float(pred_shift), 'threshold': self.pred_shift_threshold, 'drift_detected': pred_shift > self.pred_shift_threshold, 'details': shift_details}
        
        # Aggregate: require 2+ methods
        num_methods_detecting = sum(1 for m in report['methods'].values() if m.get('drift_detected', False))
        report['drift_detected'] = num_methods_detecting >= 2
        report['methods_detecting_drift'] = num_methods_detecting
        
        # --- 5. Severity assessment ---
        if report['drift_detected']:
            if pred_shift > 0.5:
                report['severity'] = 'high'
            elif ks_pval < 0.01 or psi > 0.3:
                report['severity'] = 'medium'
            else:
                report['severity'] = 'low'
        else:
            report['severity'] = 'none'
        
        # Store in history
        self.window_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'window_id': window_id,
            'drift_detected': report['drift_detected'],
            'severity': report['severity'],
            'pred_counts': new_pred_counts
        })
        
        logger.info(
            f"Drift detection: {report['drift_detected']} "
            f"(ks={ks_pval:.4f}, psi={psi:.4f}, pred_shift={pred_shift:.4f})"
        )
        
        return report
    
    def _compute_psi(self, baseline: np.ndarray, current: np.ndarray) -> float:
        """
        Compute Population Stability Index.
        """
        # Discretize into bins
        num_bins = 10
        bin_edges = np.percentile(
            np.concatenate([baseline, current]),
            np.linspace(0, 100, num_bins + 1)
        )
        bin_edges[-1] += 1e-6  # Ensure upper edge inclusive
        
        baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
        current_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Normalize
        baseline_pct = baseline_counts / baseline_counts.sum()
        current_pct = current_counts / current_counts.sum()
        
        # Compute PSI (avoid log(0))
        psi = 0.0
        for b, c in zip(baseline_pct, current_pct):
            if b > 0 and c > 0:
                psi += (c - b) * np.log(c / b)
            elif c > 0 and b == 0:
                psi += c * np.log(c / 0.001)  # Small constant
        
        return psi
    
    def _compute_prediction_shift(
        self,
        baseline_counts: Dict[int, int],
        current_counts: Dict[int, int]
    ) -> Tuple[float, Dict]:
        """
        Compute shift in prediction distribution (chi-square based).
        """
        all_classes = set(baseline_counts.keys()) | set(current_counts.keys())
        
        baseline_arr = np.array([baseline_counts.get(c, 0) for c in sorted(all_classes)])
        current_arr = np.array([current_counts.get(c, 0) for c in sorted(all_classes)])
        
        # Normalize
        baseline_pct = baseline_arr / baseline_arr.sum()
        current_pct = current_arr / current_arr.sum()
        
        # L1 distance (sum of absolute differences)
        shift = np.abs(current_pct - baseline_pct).sum() / 2.0
        
        details = {
            'baseline_distribution': dict(zip(sorted(all_classes), baseline_pct.tolist())),
            'current_distribution': dict(zip(sorted(all_classes), current_pct.tolist())),
            'class_shifts': {
                int(c): float(current_pct[i] - baseline_pct[i])
                for i, c in enumerate(sorted(all_classes))
            }
        }
        
        return shift, details
    
    @property
    def baseline(self) -> Optional[Dict]:
        """Get baseline state as a serializable dict."""
        if self.baseline_scores is None:
            return None
        
        return {
            'baseline_scores': self.baseline_scores.tolist() if hasattr(self.baseline_scores, 'tolist') else self.baseline_scores,
            'baseline_pred_counts': self.baseline_pred_counts
        }
    
    @baseline.setter
    def baseline(self, state: Dict) -> None:
        """Restore baseline state from serialized dict."""
        if state is None:
            self.baseline_scores = None
            self.baseline_pred_counts = None
        else:
            self.baseline_scores = np.asarray(state.get('baseline_scores'))
            self.baseline_pred_counts = state.get('baseline_pred_counts', {})
            logger.info(f"Baseline restored: {len(self.baseline_scores)} scores, {self.baseline_pred_counts}")
    
    def get_drift_summary(self) -> pd.DataFrame:
        """Get history of drift detections as DataFrame."""
        if not self.window_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.window_history)
    
    def save_drift_signals_to_csv(self, drift_report: Dict) -> None:
        """
        Save drift detection signals to CSV for audit and monitoring.
        """
        drift_signals_file = EVAL_DIR / "drift_signals.csv"
        
        try:
            # Extract key metrics from drift report
            signal_record = {
                'timestamp': drift_report.get('timestamp', datetime.now(timezone.utc).isoformat()),
                'drift_detected': drift_report.get('drift_detected', False),
                'severity': drift_report.get('severity', 'none'),
                'sample_count': drift_report.get('sample_count', 0),
                'methods_detecting_drift': drift_report.get('methods_detecting_drift', 0),
                'ks_statistic': drift_report.get('methods', {}).get('ks_test', {}).get('statistic', None),
                'ks_pvalue': drift_report.get('methods', {}).get('ks_test', {}).get('p_value', None),
                'ks_threshold': drift_report.get('methods', {}).get('ks_test', {}).get('threshold', None),
                'psi': drift_report.get('methods', {}).get('psi', {}).get('psi', None),
                'psi_threshold': drift_report.get('methods', {}).get('psi', {}).get('threshold', None),
                'prediction_shift': drift_report.get('methods', {}).get('prediction_shift', {}).get('shift', None),
                'prediction_shift_threshold': drift_report.get('methods', {}).get('prediction_shift', {}).get('threshold', None),
                'reason': drift_report.get('reason', '')
            }
            
            # Convert to DataFrame for easy append
            df_new = pd.DataFrame([signal_record])
            
            # Append to existing file or create new
            if drift_signals_file.exists():
                df_existing = pd.read_csv(drift_signals_file)
                df = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df = df_new
            
            df.to_csv(drift_signals_file, index=False)
            logger.info(f"Drift signal saved to {drift_signals_file}")
            
        except Exception as e:
            logger.error(f"Error saving drift signals to CSV: {e}")
