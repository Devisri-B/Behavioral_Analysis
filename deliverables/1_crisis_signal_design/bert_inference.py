"""
BERT binary classifier for suicidal ideation detection.
Loads fine-tuned model, runs inference, applies temperature scaling + thresholding.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import json
import sys
import os

logger = logging.getLogger(__name__)


class BERTRiskPredictor:
    """Binary BERT classifer: suicidal_ideation vs normal."""
    
    def __init__(self, model_dir: str, num_labels: int = 2, device: str = "auto", max_length: int = 256, batch_size: int = 32):
        self.model_dir = model_dir
        self.num_labels = num_labels
        self.max_length = max_length
        self.batch_size = batch_size
        
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else (device if device != "auto" else "cpu")
        logger.info(f"Loading BERT from {model_dir} on {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels, local_files_only=True)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load BERT: {e}")
            raise
    
    def predict_batch(self, texts: List[str], return_logits: bool = False) -> np.ndarray:
        """Batch inference returning probabilities or logits."""
        all_probs = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            batch_probs = self._predict_batch_internal(batch_texts, return_logits)
            all_probs.append(batch_probs)
        return np.vstack(all_probs)
    
    def _predict_batch_internal(self, texts: List[str], return_logits: bool = False) -> np.ndarray:
        """Internal batch inference."""
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        with torch.no_grad():
            outputs = self.model(**encodings)
        if return_logits:
            return outputs.logits.detach().cpu().numpy()
        probs = torch.softmax(outputs.logits, dim=1)
        return probs.detach().cpu().numpy()
    
    def predict_single(self, text: str) -> Tuple[float, int]:
        """Single text prediction."""
        probs = self.predict_batch([text])
        prob_suicidal = float(probs[0, 1])
        predicted_class = int(np.argmax(probs[0]))
        return prob_suicidal, predicted_class
    
    def get_suicidal_probabilities(self, texts: List[str]) -> np.ndarray:
        """Get P(suicidal) for batch of texts."""
        probs = self.predict_batch(texts)
        return probs[:, 1]


def _load_temperature_from_calibration() -> float:
    """Load temperature scaling from calibration_artifacts.json (default 1.0)."""
    current_dir = Path(__file__).parent
    calib_path = current_dir / "calibration_artifacts.json"
    try:
        if calib_path.exists():
            with open(calib_path, 'r') as f:
                calib_data = json.load(f)
            temp = float(calib_data.get('temperature', 1.0))
            logger.info(f"Loaded temperature: {temp:.4f}")
            return temp
        else:
            logger.warning(f"Calibration not found, using T=1.0")
            return 1.0
    except Exception as e:
        logger.warning(f"Failed to load calibration: {e}, using T=1.0")
        return 1.0


def apply_temperature_scaling(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to logits and return calibrated probabilities."""
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return probs


def _load_optimal_threshold_from_config() -> float:
    """
    Load Youden's J optimal threshold from governance_config.json.
    """
    # Build list of candidate paths in fallback order
    candidate_paths = []
    
    # 1. Environment variable (highest priority if set)
    env_config = os.environ.get('GOVERNANCE_CONFIG')
    if env_config:
        candidate_paths.append(Path(env_config))
        logger.debug(f"Checking GOVERNANCE_CONFIG env var: {env_config}")
    
    # 2. Module's own directory (where bert_inference.py is located)
    module_dir = Path(__file__).parent
    candidate_paths.append(module_dir / "governance_config.json")
    logger.debug(f"Checking module directory: {module_dir / 'governance_config.json'}")
    
    # 3. Current working directory
    cwd_path = Path.cwd() / "governance_config.json"
    candidate_paths.append(cwd_path)
    logger.debug(f"Checking current working directory: {cwd_path}")
    
    # Find first existing path
    config_path = None
    for path in candidate_paths:
        if path.exists():
            config_path = path
            logger.info(f"Found governance_config.json at: {config_path}")
            break
    
    if config_path is None:
        search_locations = "\n  ".join(str(p) for p in candidate_paths)
        raise FileNotFoundError(
            f"governance_config.json not found in any search location:\n  {search_locations}\n"
            "Set GOVERNANCE_CONFIG environment variable or place file in module directory or CWD."
        )
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if 'optimal_threshold' not in config:
            raise KeyError(
                "optimal_threshold not found in governance_config.json. "
                "  ERROR: Must compute Youden's J and add to config."
            )
        
        threshold = float(config['optimal_threshold'])
        
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"optimal_threshold must be in [0.0, 1.0], got {threshold}. "
                "  ERROR: Invalid threshold value."
            )
        
        logger.info(f"Loaded Youden's J optimal threshold: {threshold:.4f} from {config_path}")
        return threshold
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in governance_config.json at {config_path}: {e}")


def apply_cumulative_thresholds(
    probs: np.ndarray,
    threshold_critical: Optional[float] = None,
    apply_temp_scaling: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    # Apply Youden's J threshold to get binary escalation flags
 
    if threshold_critical is None:
        try:
            threshold_critical = _load_optimal_threshold_from_config()
        except (FileNotFoundError, KeyError, ValueError) as e:
            logger.error(f"  FAILURE: Cannot load Youden's J threshold: {e}")
            raise
    else:
        logger.info(f"Using provided threshold: {threshold_critical:.4f}")
    
    # Binary decision: above or below optimal threshold
    escalate_flags = (probs >= threshold_critical).astype(int)
    
    confidence = np.maximum(probs, 1 - probs)
    
    return escalate_flags, confidence


def risk_tier_to_string(tier: int) -> str:
    """Convert binary escalation flag (0,1) to string."""
    mapping = {0: "Normal", 1: "Suicidal_Ideation"}
    return mapping.get(tier, "Unknown")


