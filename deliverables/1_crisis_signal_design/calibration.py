"""
Temperature scaling for probability calibration.
Adjusts BERT's logits so predicted confidence matches actual accuracy.
"""

import numpy as np
from scipy.optimize import minimize

import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class TemperatureScaler:
    """Post-hoc temperature scaling for probability calibration."""
    
    def __init__(self):
        self.temperature = None
        self.is_fitted = False
    
    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        max_iter: int = 1000
    ) -> None:
        labels = np.asarray(labels, dtype=int)
        logits = np.asarray(logits, dtype=np.float32)
        
        if len(logits.shape) == 1:
            # Binary case: expand to 2D
            logits = np.column_stack([1 - logits, logits])
        
        def nll_loss(T):
            """Negative log-likelihood under temperature scaling."""
            T = np.exp(T)  # Ensure T > 0 by optimizing log(T)
            scaled_logits = logits / T
            
            # Softmax
            exp_logits = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            
            # NLL
            nll = -np.log(probs[np.arange(len(labels)), labels] + 1e-10).mean()
            return nll
        
        # Optimize
        T_init = np.log(1.0)  # Start with T=1
        result = minimize(
            nll_loss,
            T_init,
            method='Nelder-Mead',
            options={'maxiter': max_iter}
        )
        
        self.temperature = np.exp(result.x[0])
        self.is_fitted = True
        logger.info(f"Temperature scaling fitted: T = {self.temperature:.4f}")
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        # Apply temperature scaling to logits → calibrated probabilities
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        logits = np.asarray(logits, dtype=np.float32)
        
        if len(logits.shape) == 1:
            logits = np.column_stack([1 - logits, logits])
        
        # Scale and softmax
        scaled_logits = logits / self.temperature
        exp_logits = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        
        return probs


def compute_expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    """
    probs = np.asarray(probs).flatten()
    labels = np.asarray(labels, dtype=int).flatten()
    
    ece = 0.0
    bin_edges = np.linspace(0, 1, num_bins + 1)
    
    for i in range(num_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        
        if mask.sum() == 0:
            continue
        
        avg_confidence = probs[mask].mean()
        accuracy = labels[mask].mean()
        
        ece += np.abs(avg_confidence - accuracy) * mask.sum() / len(labels)
    
    return ece
