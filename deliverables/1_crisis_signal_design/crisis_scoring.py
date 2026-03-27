"""
County-level crisis scoring from post-level BERT predictions.
Combines sentiment, volume spikes, and demographic signals.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
import os

logger = logging.getLogger(__name__)


def _load_governance_config() -> Dict:
    """Load thresholds from governance_config.json (checks env var, module dir, cwd, sibling)."""
    candidate_paths = []
    
    env_config = os.environ.get('GOVERNANCE_CONFIG')
    if env_config:
        candidate_paths.append(Path(env_config))
    
    module_dir = Path(__file__).parent
    candidate_paths.append(module_dir / "governance_config.json")
    candidate_paths.append(Path.cwd() / "governance_config.json")
    candidate_paths.append(module_dir.parent / "2_governance_controls" / "governance_config.json")
    
    config_path = None
    for path in candidate_paths:
        if path.exists():
            config_path = path
            logger.info(f"Loaded governance config from {config_path}")
            break
    
    if config_path is None:
        raise FileNotFoundError(f"governance_config.json not found. Set GOVERNANCE_CONFIG env var.")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {config_path}: {e}")


@dataclass
class CrisisSignalConfig:
    """Config loaded from governance_config.json."""
    
    MIN_POSTS_COUNTY: int = None             # Min posts for county-level signal
    MIN_POSTS_WINDOW: int = None             # Min posts per time window
    
    HIGH_RISK_THRESHOLD: float = 0.70        # P(suicidal) threshold
    MEDIUM_RISK_THRESHOLD: float = None      # Monitor threshold
    BASELINE_RISK_RATE: float = 0.25         # Expected baseline rate
    
    SPIKE_MULTIPLIER: float = 2.0            # Volume spike detection multiplier
    MIN_SPIKE_POSTS: int = 10                # Minimum posts to trigger spike
    
    SMOOTHING_ALPHA: float = None            # EMA smoothing factor
    
    MIN_CONFIDENCE: float = 0.7              # Confidence gate
    STABILITY_WINDOW: int = 3                # Persistence requirement
    
    def __post_init__(self):
        """Load thresholds from governance_config.json."""
        gov_config = _load_governance_config()
        
        if self.MIN_POSTS_COUNTY is None:
            self.MIN_POSTS_COUNTY = gov_config.get("min_sample_threshold", 20)
        if self.MIN_POSTS_WINDOW is None:
            self.MIN_POSTS_WINDOW = gov_config.get("min_sample_threshold", 20)
        if self.MEDIUM_RISK_THRESHOLD is None:
            self.MEDIUM_RISK_THRESHOLD = gov_config.get("threshold_monitor", 0.50)
        if self.SMOOTHING_ALPHA is None:
            self.SMOOTHING_ALPHA = gov_config.get("ema_alpha", 0.2)
        
        logger.info(f"CrisisSignalConfig loaded: MIN_POSTS={self.MIN_POSTS_COUNTY}, "
                   f"MEDIUM_RISK={self.MEDIUM_RISK_THRESHOLD}, SMOOTHING_ALPHA={self.SMOOTHING_ALPHA}")


class CrisisScorer:
    """Computes county-level crisis signals from BERT predictions."""
    
    def __init__(self, config: Optional[CrisisSignalConfig] = None):
        self.config = config or CrisisSignalConfig()
        self.baseline_rates = {}
        self.ema_history = {}
        
    def compute_sentiment_intensity(self, predictions: np.ndarray, confidence_calibrated: bool = True) -> float:
        """Average intensity of high-risk predictions."""
        if len(predictions) == 0:
            return 0.0
        high_risk_mask = predictions >= self.config.HIGH_RISK_THRESHOLD
        if high_risk_mask.sum() == 0:
            return 0.0
        intensity = predictions[high_risk_mask].mean()
        return intensity
    
    def compute_volume_spike(self, current_count: int, baseline_count: Optional[int] = None, window_hours: int = 24) -> Tuple[float, bool]:
        """Detect volume spikes vs baseline."""
        baseline = baseline_count or int(self.config.MIN_SPIKE_POSTS / self.config.SPIKE_MULTIPLIER)
        if baseline == 0:
            baseline = 1
        spike_ratio = current_count / baseline
        is_spike = (spike_ratio >= self.config.SPIKE_MULTIPLIER and current_count >= self.config.MIN_SPIKE_POSTS)
        return spike_ratio, is_spike
    
    def compute_geographic_clustering(self, county_df: pd.DataFrame, min_counties: int = 3) -> Tuple[float, bool]:
        """Detect geographic clustering across counties."""
        if len(county_df) == 0:
            return 0.0, False
        
        # Filter counties with sufficient samples
        valid_counties = county_df[county_df['post_count'] >= self.config.MIN_POSTS_COUNTY]
        
        if len(valid_counties) == 0:
            return 0.0, False
        
        # Count elevated-risk counties
        elevated_counties = valid_counties[
            valid_counties['risk_score'] >= self.config.MEDIUM_RISK_THRESHOLD
        ]
        
        clustering_score = len(elevated_counties) / len(valid_counties)
        is_clustered = len(elevated_counties) >= min_counties
        
        logger.debug(f"Geographic clustering: {len(elevated_counties)}/{len(valid_counties)} counties elevated (score={clustering_score:.2f})")
        return clustering_score, is_clustered
    
    def compute_confidence_estimate(self, sample_size: int, predictions: np.ndarray, historical_variance: Optional[float] = None) -> float:
        """Estimate signal confidence based on sample size and variance."""
        if sample_size < self.config.MIN_POSTS_WINDOW:
            return 0.0
        
        # Compute prediction variance (higher = less confident)
        pred_var = predictions.var() if len(predictions) > 1 else 0.5
        
        # Normalize by historical variance or use default
        normalized_var = pred_var / (historical_variance or 0.1)
        
        # Confidence increases with sample size, decreases with variance
        sample_size_factor = min(1.0, sample_size / self.config.MIN_SPIKE_POSTS)
        confidence = sample_size_factor * (1 - min(1.0, normalized_var))
        
        logger.debug(f"Confidence estimate: {confidence:.3f} (N={sample_size}, var={pred_var:.3f})")
        return confidence
    
    def apply_temporal_smoothing(self, county: str, raw_score: float) -> float:
        """Apply EMA smoothing to reduce noise."""
        if county not in self.ema_history:
            self.ema_history[county] = raw_score
        
        ema = (self.config.SMOOTHING_ALPHA * raw_score + 
               (1 - self.config.SMOOTHING_ALPHA) * self.ema_history[county])
        
        self.ema_history[county] = ema
        return ema
    
    def compute_crisis_score(self, predictions: np.ndarray, county: str, volume_spike: bool, geographic_cluster: bool, sample_size: int) -> Tuple[float, Dict[str, float], bool]:
        """Compute crisis score: 40% sentiment + 30% volume + 20% geography + 10% confidence."""
        if sample_size < self.config.MIN_POSTS_WINDOW:
            return 0.0, {}, False
        
        sentiment = self.compute_sentiment_intensity(predictions)
        spike_score = float(volume_spike)
        cluster_score = float(geographic_cluster)
        confidence = self.compute_confidence_estimate(sample_size, predictions)
        
        raw_score = 0.40 * sentiment + 0.30 * spike_score + 0.20 * cluster_score + 0.10 * confidence
        smoothed_score = self.apply_temporal_smoothing(county, raw_score)
        escalate = (smoothed_score >= self.config.MEDIUM_RISK_THRESHOLD and confidence >= self.config.MIN_CONFIDENCE)
        
        component_scores = {
            'sentiment': sentiment, 'volume_spike': spike_score, 'geographic_cluster': cluster_score,
            'confidence': confidence, 'raw_score': raw_score, 'smoothed_score': smoothed_score
        }
        return smoothed_score, component_scores, escalate
