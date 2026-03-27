"""
ETL pipeline - fetches Reddit posts, runs BERT inference, geocodes, scores by county.
Stores escalations as CSV files with audit trail.
"""

import praw
import pandas as pd
import re
import os
import sys
import datetime
import time
import spacy
import uuid
from pathlib import Path
from typing import cast, Dict, List
from geopy.geocoders import Nominatim
from textblob import TextBlob
import contractions
import nltk
from nltk.corpus import stopwords
import numpy as np
import logging
import json
import pickle
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    # Import ML modules from other deliverables
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    monitoring_path = str(current_dir)
    
    crisis_path = str(project_root / 'deliverables' / '1_crisis_signal_design')
    governance_path = str(project_root / 'deliverables' / '2_governance_controls')
    
    for path in [monitoring_path, crisis_path, governance_path]:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    from bert_inference import BERTRiskPredictor, apply_cumulative_thresholds, risk_tier_to_string
    from calibration import TemperatureScaler
    from governance import GovernanceEngine, compute_youden_thresholds
    from drift_detection import DriftDetector
    from continuous_eval import ContinuousEvaluator
    
except ImportError as import_error:
    print(f"Import Error: {import_error}")
    raise

# Logging & Config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REDDIT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
SKIP_GEOCODING = os.getenv('SKIP_GEOCODING', 'false').lower() == 'true'
MAX_GEOCODING_POSTS = int(os.getenv('MAX_GEOCODING_POSTS', '100'))

# CSV storage directory
EVAL_DIR = Path('output/evaluations')
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Model paths
BERT_MODEL_DIR = os.getenv('BERT_MODEL_DIR', 'output/binary_model/final_model')
CALIBRATION_ARTIFACT = os.getenv('CALIBRATION_ARTIFACT', 'output/binary_model/calibration_artifacts.json')
GOVERNANCE_CONFIG = os.getenv('GOVERNANCE_CONFIG', 'deliverables/2_governance_controls/governance_config.json')
DRIFT_BASELINE = os.getenv('DRIFT_BASELINE', 'output/evaluations/drift_baseline.pkl')

try:
    nltk.download('stopwords', quiet=True)
    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        logger.info("Downloading spacy model 'en_core_web_md'...")
        import subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_md"], check=True)
        nlp = spacy.load("en_core_web_md")
    
# Initialize BERT predictor
    logger.info(f"Loading BERT model from {BERT_MODEL_DIR}")
    
    # Verify model directory exists
    if not os.path.exists(BERT_MODEL_DIR):
        logger.error(f"BERT model directory not found: {BERT_MODEL_DIR}")
        logger.error(f"Please ensure the model is available at: {BERT_MODEL_DIR}")
        raise FileNotFoundError(f"Model path does not exist: {BERT_MODEL_DIR}")
    
    bert_model = BERTRiskPredictor(
        model_dir=BERT_MODEL_DIR,
        num_labels=2,
        device='auto',
        max_length=256,
        batch_size=32
    )
    
    # Initialize temperature calibrator
    calibrator = TemperatureScaler()
    # Note: Load pre-fitted calibrator if artifact exists
    # For now, using uncalibrated (temperature=1.0)
    if os.path.exists(CALIBRATION_ARTIFACT):
        logger.info(f"Loading calibration artifact from {CALIBRATION_ARTIFACT}")
        with open(CALIBRATION_ARTIFACT, 'r') as f:
            calib_data = json.load(f)
            calibrator.temperature = calib_data.get('temperature', 1.0)
            calibrator.is_fitted = calib_data.get('is_fitted', False)
            if not calibrator.is_fitted:
                logger.warning("Calibration not fitted - using raw probabilities (T=1.0)")
    else:
        logger.warning(f"No calibration artifact found. Using uncalibrated probabilities.")
        calibrator.temperature = 1.0
        calibrator.is_fitted = False
    
    # Initialize governance engine
    logger.info("Initializing governance engine")
    gov_config = {}
    if os.path.exists(GOVERNANCE_CONFIG):
        with open(GOVERNANCE_CONFIG, 'r') as f:
            gov_config = json.load(f)
    
    # Use Youden's optimal threshold if available in config, otherwise default to 0.70
    optimal_threshold_value = gov_config.get('optimal_threshold', 0.70)
    
    governance_engine = GovernanceEngine(
        min_sample_threshold=gov_config.get('min_sample_threshold', 20),
        ema_alpha=gov_config.get('ema_alpha', 0.2),
        threshold_monitor=gov_config.get('threshold_monitor', 0.5),
        threshold_escalate=gov_config.get('threshold_escalate', 0.75),
        confidence_min=gov_config.get('confidence_min', 0.6),
        persistence_windows=gov_config.get('persistence_windows', 2),
        optimal_threshold=optimal_threshold_value,  # Data-driven from Youden's J
    )
    
    # Initialize drift detector and load baseline if available
    drift_detector = DriftDetector(
        ks_threshold=0.05,
        psi_threshold=0.15,
        pred_shift_threshold=0.20
    )
    
    # Load baseline from pickle if it exists (to enable drift detection on subsequent runs)
    baseline_loaded = False
    if os.path.exists(DRIFT_BASELINE):
        try:
            with open(DRIFT_BASELINE, 'rb') as f:
                baseline_state = pickle.load(f)
                drift_detector.baseline = baseline_state
                baseline_loaded = True
                logger.info(f"Loaded drift baseline from {os.path.abspath(DRIFT_BASELINE)}")
        except Exception as e:
            logger.warning(f"Could not load drift baseline: {e}")
            baseline_loaded = False
    else:
        logger.debug(f"No drift baseline found at {DRIFT_BASELINE} (first run - will initialize on successful completion)")
    
    # Initialize continuous evaluator
    evaluator = ContinuousEvaluator(
        min_reviewed_samples=100,
        min_new_samples_for_retrain=50
    )
    
except Exception as e:
    logger.error(f"Model Loading Failed: {e}")
    raise

geolocator = Nominatim(user_agent="crisis_monitor_bert_v1")
stop = set(stopwords.words('english')) - {"not", "no"}
SUBREDDITS = [
    'mentalhealth', 'depression', 'SuicideWatch', 'anxiety', 'stress',
    'offmychest', 'lonely', 'BPD', 'ptsd', 'socialanxiety', 'bipolar',
    'addiction', 'traumatoolbox', 'CPTSD', 'selfharm', 'OCD',
    'EatingDisorders', 'MentalHealthSupport', 'schizophrenia',
    'insomnia', 'panicattacks', 'ADHD'
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _extract_sentiment_polarity(text: str) -> float:
    """Extract sentiment polarity using TextBlob: -1 (negative) to 1 (positive)."""
    try:
        blob = TextBlob(str(text))
        polarity = cast(float, blob.polarity)
        return polarity
    except Exception as e:
        logger.warning(f"Failed to extract sentiment: {e}")
        return 0.0

def get_reddit():
    """Fetch latest posts from crisis-related subreddits."""
    if not REDDIT_ID:
        logger.warning("REDDIT_CLIENT_ID not set")
        return pd.DataFrame()
    
    reddit = praw.Reddit(
        client_id=REDDIT_ID,
        client_secret=REDDIT_SECRET,
        user_agent='crisis_monitor_bert_v1'
    )
    
    posts = []
    for sub in SUBREDDITS:
        try:
            for post in reddit.subreddit(sub).new(limit=50):
                posts.append({
                    'id': post.id,
                    'created_utc': datetime.datetime.fromtimestamp(post.created_utc),
                    'subreddit': sub,
                    'text': f"{post.title} {post.selftext}"[:2000],
                    'url': post.url
                })
        except Exception as e:
            logger.warning(f"Error fetching from r/{sub}: {e}")
    
    return pd.DataFrame(posts)


def clean_for_model(text: str) -> str:
    """Clean text: lowercase, expand contractions, remove URLs."""
    text_str = str(text).lower()
    fixed_text = contractions.fix(text_str)
    text_str = str(fixed_text) if fixed_text is not None else text_str
    text_str = re.sub(r'http\S+', '', text_str)
    return text_str


def save_escalation_batch(records: List[Dict]):
    """Append records to appropriate CSV files based on status: suicidal_detection.csv or non_suicidal_detection.csv."""
    total_saved = 0
    
    try:
        # Separate records by status
        suicidal_records = [r for r in records if r.get('status') == 'Suicidal']
        non_suicidal_records = [r for r in records if r.get('status') == 'Non-Suicidal']
        
        # Save suicidal detections
        if suicidal_records:
            suicidal_file = EVAL_DIR / "suicidal_detection.csv"
            df_new = pd.DataFrame(suicidal_records)
            if suicidal_file.exists():
                df_existing = pd.read_csv(suicidal_file)
                df = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df = df_new
            df.to_csv(suicidal_file, index=False)
            logger.debug(f"Saved {len(suicidal_records)} suicidal detection records to suicidal_detection.csv")
            total_saved += len(suicidal_records)
        
        # Save non-suicidal detections
        if non_suicidal_records:
            non_suicidal_file = EVAL_DIR / "non_suicidal_detection.csv"
            df_new = pd.DataFrame(non_suicidal_records)
            if non_suicidal_file.exists():
                df_existing = pd.read_csv(non_suicidal_file)
                df = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df = df_new
            df.to_csv(non_suicidal_file, index=False)
            logger.debug(f"Saved {len(non_suicidal_records)} non-suicidal detection records to non_suicidal_detection.csv")
            total_saved += len(non_suicidal_records)
        
        return total_saved
    except Exception as e:
        logger.error(f"Error saving records: {e}")
        return 0


def log_etl_execution(status: str, events_processed: int, events_escalated: int, error_msg: str = None):
    """Log ETL pipeline execution to CSV."""
    etl_logs_file = EVAL_DIR / "etl_logs.csv"
    try:
        log_record = {
            'run_timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'status': status,
            'events_processed': events_processed,
            'events_escalated': events_escalated,
            'error_message': error_msg,
            'duration_seconds': 0
        }
        df_new = pd.DataFrame([log_record])
        if etl_logs_file.exists():
            df_existing = pd.read_csv(etl_logs_file)
            df = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df = df_new
        df.to_csv(etl_logs_file, index=False)
        logger.info("ETL execution logged to CSV")
        return True
    except Exception as e:
        logger.error(f"Error logging ETL execution: {e}")
        return False

def filter_existing_posts(df):
    # Remove in-batch duplicates (database handles rest via UPSERT)
    if df.empty:
        return df
    
    # Remove intra-batch duplicates (same post fetched multiple times in this batch)
    duplicates = df['id'].duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates(subset=['id'], keep='first')
        logger.info(f"   - Removed {duplicates} duplicate texts in batch")
    
    # Note: Database-level deduplication is handled by INSERT ... ON CONFLICT (UPSERT)
    # So I don't filter against existing IDs here - let the database handle it
    logger.info(f"   - Processing {len(df)} posts (UPSERT will handle any existing IDs)")
    return df


def infer_bert_risk(texts):
    # BERT inference + temperature scaling calibration
    try:
        # Batch inference
        logits = bert_model.predict_batch(texts, return_logits=True)
        
        # Apply temperature scaling
        if calibrator.is_fitted:
            probs = calibrator.calibrate(logits)
        else:
            # Uncalibrated softmax
            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        
        return probs
    except Exception as e:
        logger.error(f"BERT inference failed: {e}")
        raise


def create_audit_fields(pred_id, probs, tier, conf, model_version, threshold_version):
    """Create structured audit trail for prediction.
    
    Args:
        pred_id: Unique prediction identifier
        probs: 2-element array [P(Non-Suicidal), P(Suicidal)] from binary classifier
        tier: Predicted binary tier (0=Non-Suicidal, 1=Suicidal)
        conf: Model confidence (max probability across classes)
        model_version: Version of BERT model used
        threshold_version: Version of thresholding rules applied
        
    Returns:
        dict: Audit trail with prediction components and governance metadata
    """
    return {
        'pred_id': pred_id,
        'model_version': model_version,
        'threshold_version': threshold_version,
        'prob_non_suicidal': float(probs[0]),
        'prob_suicidal': float(probs[1]),
        'predicted_tier': int(tier),
        'confidence': float(conf),
        'audit_timestamp': datetime.datetime.utcnow().isoformat()
    }



def run_pipeline():
    """Main ETL pipeline."""
    logger.info("=" * 80)
    logger.info("Starting ETL Pipeline")
    logger.info("=" * 80)
    
    # --- 1. Fetch Reddit posts ---
    logger.info("1. Extracting Reddit posts...")
    df = get_reddit()
    if df.empty:
        logger.warning("No posts fetched")
        return
    
    # Deduplicate in batch
    initial_count = len(df)
    df = df.drop_duplicates(subset=['text'])
    logger.info(f"   - Removed {initial_count - len(df)} duplicate texts in batch")
    
    # Filter existing posts
    df = filter_existing_posts(df)
    if df.empty:
        logger.info("   - No new posts")
        return
    
    logger.info(f"   - Processing {len(df)} new posts")
    
    # --- 2. Clean text and run BERT inference ---
    logger.info("2. Running BERT inference...")
    df['clean_text'] = df['text'].apply(clean_for_model)
    
    # BERT inference
    try:
        bert_probs = infer_bert_risk(df['clean_text'].tolist())
    except Exception as e:
        logger.error(f"BERT inference failed: {e}")
        return
    
    # --- Risk Tier Assignment (uses apply_cumulative_thresholds, the authoritative function) ---
    suicidal_probs = bert_probs[:, 1]  # Extract P(Suicidal)
    tiers, _ = apply_cumulative_thresholds(suicidal_probs, threshold_critical=None)
    
    # CRITICAL: Confidence measures model certainty across BOTH classes, not just positive class
    confidences = np.max(bert_probs, axis=1)
    
    # Human-in-the-loop escalation gate 
    review_threshold = 0.60
    needs_review = confidences < review_threshold
    
    df['prob_non_suicidal'] = bert_probs[:, 0]
    df['prob_suicidal'] = bert_probs[:, 1]
    df['risk_tier'] = tiers
    df['confidence'] = confidences
    df['needs_human_review'] = needs_review
    
    # TextBlob sentiment analysis (auxiliary signal)
    df['sentiment'] = df['text'].apply(_extract_sentiment_polarity)
    
    logger.info(f"   - Risk distribution - Non-Suicidal: {(tiers == 0).sum()}, Suicidal: {(tiers == 1).sum()}")
    logger.info(f"   - Escalated for human review: {needs_review.sum()}")
    
    # Extract suicidal probability scores for drift detection
    suicidal_scores = bert_probs[:, 1]
    
    # --- 3. Geolocation ---
    logger.info("3. Geolocating posts...")
    
    if SKIP_GEOCODING:
        logger.info("   - Geocoding disabled (SKIP_GEOCODING=true)")
        df['location_name'] = None
        df['lat'] = None
        df['lon'] = None
    else:
        # Geocode ALL posts (Nominatim rate-limited to ~1 req/sec, so will take time)
        posts_to_geocode = len(df)
        logger.info(f"   - Geocoding all {posts_to_geocode} posts (spaCy NER + Nominatim with caching)")
        logger.info(f"   - Note: This will take ~{posts_to_geocode // 60} minutes due to rate limiting")
        
        geo_results = []
        geocoding_cache = {}  # Cache to avoid re-geocoding same location
        
        for i, text in enumerate(df['text']):
            # Extract ALL locations using spaCy NER (GPE entities)
            locations = governance_engine.extract_locations_from_text(text)
            coords = None
            location_name = None
            
            if locations:
                # Try each extracted location (try all, not just first)
                for loc in locations:
                    # Check cache first
                    if loc in geocoding_cache:
                        coords, location_name = geocoding_cache[loc]
                        break
                    
                    # Try to geocode this location
                    coords = governance_engine.geocode_location(loc)
                    if coords:
                        location_name = loc
                        geocoding_cache[loc] = (coords, location_name)  # Cache success
                        break
            
            if coords:
                lat, lon = coords
                geo_results.append((location_name, lat, lon))
            else:
                geo_results.append((None, None, None))
            
            # Add inter-request delay to further prevent rate limiting
            # This adds delay BETWEEN posts, on top of delays within geolocate_post
            if i < posts_to_geocode - 1:
                time.sleep(2)  # 2 second delay between post geocoding attempts
            
            if (i + 1) % 10 == 0:
                logger.debug(f"   - Geocoded {i + 1}/{posts_to_geocode} posts")
        
        # Unpack the 3-tuple: (location_name, latitude, longitude)
        df['location_name'] = [res[0] for res in geo_results]
        df['lat'] = [res[1] for res in geo_results]
        df['lon'] = [res[2] for res in geo_results]
        

    
    geolocated = df['lat'].notna().sum()
    logger.info(f"   - Successfully geolocated {geolocated}/{len(df)} posts")
    

    # --- 3b. Low-Confidence Escalation (Human-in-the-Loop) ---
    logger.info("3b. Saving low-confidence predictions to human_review CSV...")
    human_review_file = Path('output/evaluations') / 'human_review.csv'
    low_conf_indices = df[needs_review].index
    low_conf_count = 0
    
    if len(low_conf_indices) > 0:
        human_review_data = []
        for idx in low_conf_indices:
            row = df.loc[idx]
            human_review_data.append({
                'pred_id': f"post_{row['id']}",
                'predicted_class': int(row['risk_tier']),
                'confidence': float(row['confidence']),
                'post_content': row['text'],
                'reason': f"Low model confidence ({row['confidence']:.1%}) - requires human verification",
                'priority': 'high' if row['risk_tier'] == 1 else 'normal',
                'escalation_timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()
            })
            low_conf_count += 1
        
        # Save to CSV
        try:
            df_human_review = pd.DataFrame(human_review_data)
            if human_review_file.exists():
                df_existing = pd.read_csv(human_review_file)
                df_human_review = pd.concat([df_existing, df_human_review], ignore_index=True)
            df_human_review.to_csv(human_review_file, index=False)
        except Exception as e:
            logger.error(f"Error saving human review escalations to CSV: {e}")
        
        logger.info(f"   - Saved {low_conf_count} low-confidence predictions to human_review.csv")
    
    # --- 5. Drift Detection ---
    logger.info("5. Running drift detection...")

    # This ensures first run data is saved as baseline, and drift detection begins on subsequent runs
    if not baseline_loaded:
        try:
            drift_detector.fit_baseline(suicidal_scores, tiers)
            baseline_path = os.path.abspath(DRIFT_BASELINE)
            with open(baseline_path, 'wb') as f:
                pickle.dump(drift_detector.baseline, f)
            logger.info("First run: baseline established, drift detection starts next run")
            logger.info(f"   Baseline: {len(suicidal_scores)} scores, {len(tiers)} predictions saved to {baseline_path}")
            drift_report = {}  # No drift comparison on first run
        except Exception as e:
            logger.warning(f"Could not persist drift baseline: {e}")
            drift_report = {}
    else:
        # Baseline loaded - run drift detection against established baseline
        drift_report = drift_detector.detect_drift(
            scores=suicidal_scores,
            predictions=tiers
        )
    
    # Report drift detection results (only populated if baseline_loaded=True)
    if drift_report.get('drift_detected'):
        logger.warning(f"   - DRIFT DETECTED: {drift_report['severity']}")
        logger.warning(f"     K-S p-value: {drift_report['methods']['ks_test']['p_value']:.4f}")
        logger.warning(f"     PSI: {drift_report['methods']['psi']['psi']:.4f}")
    else:
        if baseline_loaded:
            logger.info("   - No drift detected")
        # (if not baseline_loaded, drift_report is empty and this message is skipped)
    
    # Save drift signals to CSV
    if baseline_loaded and drift_report:
        drift_detector.save_drift_signals_to_csv(drift_report)
    
    # Log baseline status for monitoring
    if baseline_loaded:
        logger.debug(f"Drift detection active: baseline established, comparing current distribution")
    else:
        logger.debug(f"Drift detection inactive: baseline just created on this run")
    
    # --- 6. Prepare DB insertion ---
    logger.info("6. Preparing database insert...")
    
    # Create status string (binary: Non-Suicidal or Suicidal)
    status_map = {0: 'Non-Suicidal', 1: 'Suicidal'}
    df['status'] = df['risk_tier'].map(status_map)
    
    # Create risk factors explanation
    def create_risk_explanation(row):
        explanations = []
        if row['risk_tier'] == 1:
            explanations.append(f"Suicidal pattern ({row['prob_suicidal']:.1%})")
        if row['sentiment'] < -0.3:
            explanations.append(f"Negative sentiment ({row['sentiment']:.2f})")
        if row['needs_human_review']:
            explanations.append(f"Low confidence ({row['confidence']:.1%})")
        return " + ".join(explanations) if explanations else "Non-Suicidal"
    
    df['risk_factors'] = df.apply(create_risk_explanation, axis=1)
    
    # Impact score computation (for internal use, not persisted)
    base_scores = {'Non-Suicidal': 20, 'Suicidal': 80}
    df['impact_score'] = df['status'].map(base_scores) + (df['confidence'] * 10) + (df['sentiment'] * -5)
    df['impact_score'] = df['impact_score'].clip(0, 100)
    
    # Model metadata
    model_version = "bert_v1"
    
    df['model_version'] = model_version
    
    # --- 7. Save to CSV ---
    logger.info("7. Saving to CSV storage...")
    
    # Select columns for storage
    cols_to_save = [
        'id', 'created_utc', 'subreddit', 'text', 'status', 'sentiment',
        'risk_factors', 'url', 'location_name', 'lat', 'lon',
        'confidence', 'model_version'
    ]
    
    df_to_save = df[cols_to_save].copy()
    
    # Convert Reddit post IDs to valid UUIDs
    # Reddit IDs are alphanumeric strings, database expects UUID format
    df_to_save['id'] = df_to_save['id'].apply(
        lambda x: str(uuid.uuid5(uuid.NAMESPACE_URL, f"reddit.com/{str(x)}"))
    )
    
    if not df_to_save.empty:
        try:
            # Convert DataFrame to list of dictionaries
            records = df_to_save.to_dict('records')
            
            # Insert in batches
            batch_size = 100
            total_saved = 0
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                batch_count = save_escalation_batch(batch)
                total_saved += batch_count
                logger.info(f"   - Saved batch {i//batch_size + 1} ({len(batch)} records) to suicidal/non_suicidal CSV files")
            
            logger.info(f"   - Successfully saved {total_saved} rows (suicidal_detection.csv + non_suicidal_detection.csv)")
        except Exception as e:
            logger.error(f"   - CSV storage error: {e}")
    
    # --- 8. Log ETL execution ---
    logger.info("8. Logging ETL execution...")
    events_escalated = int((df['status'] == 'Suicidal').sum()) if not df.empty else 0
    log_etl_execution(status='success', events_processed=len(df), events_escalated=events_escalated)
    
    logger.info("=" * 80)
    logger.info("ETL Pipeline Complete")
    logger.info("=" * 80)


if __name__ == '__main__':
    run_pipeline()
