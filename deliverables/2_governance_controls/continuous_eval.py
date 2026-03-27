"""
Continuous evaluation - tracks model performance from reviews, signals retraining eligible.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import logging

logger = logging.getLogger(__name__)

# CSV storage directory for evaluation data
EVAL_DIR = Path('output/evaluations')
EVAL_DIR.mkdir(parents=True, exist_ok=True)


class ContinuousEvaluator:
    """Track model performance from human-reviewed predictions."""
    
    def __init__(self, min_reviewed_samples: int = 100, min_new_samples_for_retrain: int = 50, f1_improvement_threshold: float = 0.02, retraining_freq_days: int = 7):
        self.min_reviewed_samples = min_reviewed_samples
        self.min_new_samples_for_retrain = min_new_samples_for_retrain
        self.f1_improvement_threshold = f1_improvement_threshold
        self.retraining_freq_days = retraining_freq_days
        self.reviews = []
        self.performance_history = []
        self.last_retrain_timestamp = None
        logger.info(f"ContinuousEvaluator initialized")
    
    def _save_review_to_csv(self, review: Dict) -> None:
        """Append single review to analyst_reviews.csv."""
        analyst_reviews_file = EVAL_DIR / "analyst_reviews.csv"
        try:
            df_new = pd.DataFrame([review])
            if analyst_reviews_file.exists():
                df_existing = pd.read_csv(analyst_reviews_file)
                df = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df = df_new
            df.to_csv(analyst_reviews_file, index=False)
        except Exception as e:
            logger.error(f"Error saving review to CSV: {e}")
    
    def _save_performance_to_csv(self, metrics: Dict) -> None:
        """Append performance metrics to performance_history.csv."""
        perf_file = EVAL_DIR / "performance_history.csv"
        try:
            df_new = pd.DataFrame([metrics])
            if perf_file.exists():
                df_existing = pd.read_csv(perf_file)
                df = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df = df_new
            df.to_csv(perf_file, index=False)
        except Exception as e:
            logger.error(f"Error saving performance metrics to CSV: {e}")
    
    def add_review(self, pred_id: str, predicted_class: int, confidence: float, true_label: int, model_version: str = "unknown", threshold_version: str = "unknown", review_timestamp: Optional[str] = None) -> None:
        """Add a reviewed prediction result and save to CSV."""
        review = {
            'pred_id': pred_id,
            'model_version': model_version,
            'threshold_version': threshold_version,
            'predicted_class': int(predicted_class),
            'confidence': float(confidence),
            'true_label': int(true_label),
            'review_timestamp': review_timestamp or datetime.utcnow().isoformat(),
            'correct': int(predicted_class) == int(true_label)
        }
        
        self.reviews.append(review)
        self._save_review_to_csv(review)
        logger.debug(f"Added review: pred_id={pred_id}, correct={review['correct']}")
    
    def compute_metrics(
        self,
        model_version: Optional[str] = None,
        num_classes: int = 3
    ) -> Optional[Dict]:
        """
        Compute performance metrics for a model version.
        """
        if len(self.reviews) < self.min_reviewed_samples:
            logger.warning(
                f"Only {len(self.reviews)} reviewed samples. "
                f"Need {self.min_reviewed_samples} for reliable metrics."
            )
            return None
        
        # Filter by model version if specified
        if model_version:
            relevant = [r for r in self.reviews if r['model_version'] == model_version]
        else:
            relevant = self.reviews
        
        if len(relevant) < self.min_reviewed_samples:
            return None
        
        df = pd.DataFrame(relevant)
        
        y_pred = df['predicted_class'].values
        y_true = df['true_label'].values
        
        # Accuracy
        accuracy = (y_pred == y_true).mean()
        
        # Per-class metrics
        p, r, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            zero_division=0
        )
        
        # Macro-F1
        macro_f1 = np.mean(f1)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        
        # Calibration: bin predictions by confidence
        conf_bins = pd.cut(df['confidence'], bins=10)
        calibration_data = []
        for bin_label, group in df.groupby(conf_bins, observed=True):
            if len(group) > 0:
                bin_accuracy = group['correct'].mean()
                bin_confidence = group['confidence'].mean()
                calibration_data.append({
                    'bin': str(bin_label),
                    'n': len(group),
                    'accuracy': bin_accuracy,
                    'avg_confidence': bin_confidence,
                    'calibration_error': abs(bin_accuracy - bin_confidence)
                })
        
        calibration_df = pd.DataFrame(calibration_data)
        mean_calibration_error = calibration_df['calibration_error'].mean()
        
        metrics = {
            'model_version': model_version or 'all',
            'num_reviewed': len(relevant),
            'accuracy': float(accuracy),
            'precision_per_class': [float(x) for x in p],
            'recall_per_class': [float(x) for x in r],
            'f1_per_class': [float(x) for x in f1],
            'macro_f1': float(macro_f1),
            'confusion_matrix': cm.tolist(),
            'mean_calibration_error': float(mean_calibration_error),
            'calibration_bins': calibration_df.to_dict('records'),
            'compute_timestamp': datetime.utcnow().isoformat()
        }
        
        # Store in history and save to CSV
        self.performance_history.append(metrics)
        self._save_performance_to_csv(metrics)
        
        logger.info(
            f"Computed metrics for {model_version or 'all'}: "
            f"accuracy={accuracy:.4f}, macro_f1={macro_f1:.4f}"
        )
        
        return metrics
    
    def check_retraining_eligibility(
        self,
        baseline_f1: float = 0.80
    ) -> Dict:
        """
        Determine if retraining is eligible and recommended.
        """
        result = {
            'eligible': False,
            'recommended': False,
            'reasons': []
        }
        
        # Check 1: Sufficient reviewed samples
        if len(self.reviews) < self.min_new_samples_for_retrain:
            result['reasons'].append(
                f"Insufficient reviewed samples: {len(self.reviews)} / {self.min_new_samples_for_retrain}"
            )
        else:
            result['eligible'] = True
            result['reasons'].append(
                f"Sufficient reviewed samples: {len(self.reviews)}"
            )
        
        # Check 2: Time since last retraining
        if self.last_retrain_timestamp:
            days_since = (datetime.utcnow() - datetime.fromisoformat(self.last_retrain_timestamp)).days
            if days_since < self.retraining_freq_days:
                result['reasons'].append(
                    f"Recent retraining ({days_since} days ago). "
                    f"Wait {self.retraining_freq_days - days_since} more days."
                )
                result['eligible'] = False
            else:
                result['reasons'].append(f"Previous retraining {days_since} days ago")
        
        # Check 3: Performance improvement potential
        if result['eligible'] and len(self.reviews) >= self.min_reviewed_samples:
            current_metrics = self.compute_metrics()
            
            if current_metrics and current_metrics['macro_f1'] < baseline_f1:
                f1_gap = baseline_f1 - current_metrics['macro_f1']
                result['reasons'].append(
                    f"F1 below baseline by {f1_gap:.4f} (thres={self.f1_improvement_threshold:.4f})"
                )
                
                if f1_gap > self.f1_improvement_threshold:
                    result['recommended'] = True
                    result['reasons'].append(
                        f"F1 gap exceeds threshold. Retraining recommended."
                    )
            else:
                result['reasons'].append(
                    f"Model performing well (F1={current_metrics['macro_f1']:.4f})"
                )
        
        logger.info(
            f"Retraining eligibility: eligible={result['eligible']}, "
            f"recommended={result['recommended']}"
        )
        
        return result
    
    def record_retraining_attempt(self) -> None:
        """Record that retraining was performed."""
        self.last_retrain_timestamp = datetime.utcnow().isoformat()
        logger.info(f"Retraining recorded at {self.last_retrain_timestamp}")
    
    def get_performance_report(self) -> pd.DataFrame:
        """Get performance history as DataFrame."""
        if not self.performance_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.performance_history)
    
    def export_reviews_for_retraining(
        self,
        output_csv: str,
        model_version: Optional[str] = None
    ) -> None:
        """
        Export reviewed predictions as CSV for model retraining.
        """
        if model_version:
            relevant = [r for r in self.reviews if r['model_version'] == model_version]
        else:
            relevant = self.reviews
        
        df = pd.DataFrame(relevant)
        df = df[['pred_id', 'true_label', 'predicted_class', 'confidence', 'model_version', 'review_timestamp']]
        df = df.rename(columns={'true_label': 'label', 'predicted_class': 'pred'})
        
        df.to_csv(output_csv, index=False)
        logger.info(f"Exported {len(df)} reviewed predictions to {output_csv}")


