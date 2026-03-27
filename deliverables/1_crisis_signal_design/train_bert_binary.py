"""
Train Binary BERT for Suicidal Ideation Detection: Simple binary classification: Suicidal vs Non-Suicidal
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import Counter

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import calibration module (location relative to where script runs)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration import TemperatureScaler


def load_and_prepare_data(csv1_path: str, csv2_path: str, val_split: float = 0.1, test_split: float = 0.1):
    """
    Load two datasets and prepare binary labels (suicidal vs non-suicidal).
    """
    
    def load_single_csv(csv_path):
        """Load CSV and auto-detect text and label columns."""
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Create lowercase column mapping for case-insensitive detection
        cols_lower = {col.lower(): col for col in df.columns}
        
        # Detect label column (case-insensitive)
        label_col = None
        for col in ['class', 'label', 'sentiment', 'status', 'target', 'category', 'suicide']:
            if col in cols_lower:
                label_col = cols_lower[col]
                break
        
        text_col = None
        for col in ['text', 'content', 'post', 'statement', 'message', 'tweet', 'title']:
            if col in cols_lower:
                text_col = cols_lower[col]
                break
        
        if text_col is None or label_col is None:
            raise ValueError(f"Could not detect text and label columns in {csv_path}. Found: {df.columns.tolist()}")
        
        logger.info(f"  Text column: {text_col}, Label column: {label_col}")
        
        # Prepare data with BINARY labels
        data = []
        for idx, row in df.iterrows():
            text = str(row[text_col]).strip()
            orig_label = str(row[label_col]).lower().strip()
            
            # Map to binary: 0 = non-suicidal, 1 = suicidal
            if 'not' in orig_label or 'non-' in orig_label or 'non ' in orig_label or orig_label == '0':
                label = 0  # Non-suicidal
            elif 'suicidal' in orig_label or 'suicide' in orig_label or orig_label == '1':
                label = 1  # Suicidal
            else:
                # Unknown label - skip
                continue
            
            if len(text) > 10:  # Filter very short texts
                data.append({'text': text, 'label': label})
        
        logger.info(f"  Loaded {len(data)} samples")
        return pd.DataFrame(data)
    
    # Load both datasets
    df1 = load_single_csv(csv1_path)
    df2 = load_single_csv(csv2_path)
    
    # Combine
    combined_df = pd.concat([df1, df2], ignore_index=True)
    logger.info(f"Combined dataset size: {len(combined_df)}")
    logger.info(f"Label distribution: {combined_df['label'].value_counts().to_dict()}")
    
    # Split into train/val/test
    train_df, temp_df = train_test_split(
        combined_df, 
        test_size=(val_split + test_split),
        random_state=42,
        stratify=combined_df['label']
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_split / (val_split + test_split),
        random_state=42,
        stratify=temp_df['label']
    )
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info(f"Label distribution (train): {train_df['label'].value_counts().to_dict()}")
    logger.info(f"Label distribution (val): {val_df['label'].value_counts().to_dict()}")
    logger.info(f"Label distribution (test): {test_df['label'].value_counts().to_dict()}")
    
    return train_df, val_df, test_df


def build_datasets(train_df, val_df, model_name: str = 'bert-base-uncased', max_length: int = 256):
    """Build tokenized datasets."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)
    
    train_ds = Dataset.from_pandas(train_df[['text', 'label']])
    val_ds = Dataset.from_pandas(val_df[['text', 'label']])
    
    train_ds = train_ds.map(tokenize_function, batched=True)
    val_ds = val_ds.map(tokenize_function, batched=True)
    
    return tokenizer, train_ds, val_ds


def compute_metrics(eval_preds):
    """Compute classification metrics."""
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    p, r, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    
    try:
        roc_auc = roc_auc_score(labels, logits[:, 1])
    except:
        roc_auc = float('nan')
    
    return {
        'accuracy': accuracy,
        'precision': p,
        'recall': r,
        'f1': f1,
        'roc_auc': roc_auc
    }


class WeightedTrainer(Trainer):
    """Trainer with class-weighted loss for imbalanced data."""
    
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if self.class_weights is None:
            loss_fct = torch.nn.CrossEntropyLoss()
        else:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv1', required=True, help='Path to first dataset CSV')
    parser.add_argument('--csv2', required=True, help='Path to second dataset CSV')
    parser.add_argument('--model', default='bert-base-uncased', help='Base model name')
    parser.add_argument('--epochs', type=int, default=5)  
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_class_weight', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--output_dir', default='results/binary_model')
    parser.add_argument('--resume_from', default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load and prepare data
    train_df, val_df, test_df = load_and_prepare_data(args.csv1, args.csv2)
    
    # Build datasets and tokenizer
    tokenizer, train_ds, val_ds = build_datasets(
        train_df,
        val_df,
        args.model,
        args.max_length
    )
    
    # Load model for binary classification (2 classes)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    
    # Compute class weights for binary classification
    class_weights = None
    if args.use_class_weight:
        label_counts = train_df['label'].value_counts()
        total = label_counts.sum()
        weights = torch.tensor([total / label_counts[i] if i in label_counts else 1.0 for i in range(2)], dtype=torch.float32)
        weights = weights / weights.sum() * 2  # Normalize
        class_weights = weights
        logger.info(f"Class weights: {class_weights.tolist()}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, 'checkpoints'),
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=100,
        seed=args.seed,
        report_to=[],
        fp16=args.fp16,
    )
    
    # Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights
    )
    
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_ds = Dataset.from_pandas(test_df[['text', 'label']])
    test_ds = test_ds.map(
        lambda examples: tokenizer(examples['text'], padding='max_length', truncation=True, max_length=args.max_length),
        batched=True
    )
    
    results = trainer.evaluate(test_ds)
    logger.info(f"Test metrics: {results}")
    
    # Save final model
    model_save_path = os.path.join(args.output_dir, 'final_model')
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    logger.info(f"Model saved to {model_save_path}")
    
    # Save results
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    # Fit temperature scaling calibration on validation set
    logger.info("Fitting temperature scaling calibration on validation set...")
    val_predictions = trainer.predict(val_ds)
    val_logits = val_predictions.predictions
    val_labels = val_predictions.label_ids
    
    calibrator = TemperatureScaler()
    calibrator.fit(val_logits, val_labels)
    
    # Compute calibrated probabilities
    calibrated_probs_2d = calibrator.calibrate(val_logits)
    calibrated_probs = calibrated_probs_2d[:, 1]  # P(suicidal)
    
    # Compute Youden's J optimal threshold on calibrated probabilities
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '2_governance_controls'))
    from governance import compute_youden_thresholds
    
    optimal_threshold, sensitivity, specificity = compute_youden_thresholds(
        calibrated_probs, val_labels, num_thresholds=100
    )
    logger.info(f"Youden's J Threshold: {optimal_threshold:.4f}")
    logger.info(f"  Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
    
    # Save calibration artifact
    calib_artifact_path = os.path.join(args.output_dir, 'calibration_artifacts.json')
    calib_data = {
        'temperature': float(calibrator.temperature),
        'is_fitted': True,
        'calibration_date': str(pd.Timestamp.now()),
        'validation_samples': len(val_labels),
        'youden_threshold': float(optimal_threshold),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity)
    }
    with open(calib_artifact_path, 'w') as f:
        json.dump(calib_data, f, indent=2)
    logger.info(f"Calibration artifact saved to {calib_artifact_path}")
    
    # Update governance_config.json with new threshold
    try:
        gov_config_path = os.path.join(os.path.dirname(__file__), '..', '2_governance_controls', 'governance_config.json')
        if os.path.exists(gov_config_path):
            with open(gov_config_path, 'r') as f:
                gov_data = json.load(f)
            gov_data['optimal_threshold'] = float(optimal_threshold)
            with open(gov_config_path, 'w') as f:
                json.dump(gov_data, f, indent=2)
            logger.info(f"Updated governance_config.json with threshold: {optimal_threshold:.4f}")
    except Exception as e:
        logger.warning(f"Could not update governance_config.json: {e}")


if __name__ == '__main__':
    main()
