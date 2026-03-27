# Monitoring & Integration

**End-to-end ETL pipeline that operationalizes Deliverables 1 & 2 on real-world public health data.**

## Pipeline Architecture
```
Train BERT
    ↓
Reddit API (21 mental health subreddits)
    ↓
FETCH: 50 posts per subreddit (~1,050 posts per execution)
    ↓
CLEAN: Remove URLs, expand contractions, deduplicate
    ↓
INFER: BERT batch prediction (checkpoint-3995) with calibration
    ↓
LOCATE: Extract county FIPS via spaCy NER + Census API
    ↓
SCORE: Apply crisis scoring (Deliverable 1) + confidence gates
    ↓
GOVERN: Apply safeguards + risk tier assignment (Deliverable 2)
    ↓
ESCALATE: Route to human review or archive
    ↓
PERSIST: Store escalations & logs to CSV (output/evaluations/) for reproducibility
    ↓
MONITOR: Track drift, log health metrics, alert on anomalies

--- 

## Implementation Details

### 1. Training the BERT Model

**Note**: BERT fine-tuning to enable the downstream governance and monitoring stages. 

Datasets used to train are 
1. Suicidal Ideation Detection Reddit Dataset, https://doi.org/10.17632/z8s6w86tr3.2
2. Kaggle Suicide and Depression Detection, https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch/data

#### To Train BERT 

```bash
python3 train_bert_binary.py \
  --csv1 path/to/reddit_data.csv \
  --csv2 path/to/twitter_data.csv \
  --output_dir output/binary_model \
  --epochs 5 \
  --batch_size 16
```

#### Generated Output (from `train_bert_binary.py`)

| File/Folder | Purpose |
|------|---------|
| `output/binary_model/calibration_artifacts.json` | Calibration state: temperature, threshold, sensitivity, specificity |
| `output/binary_model/final_model/` | Final trained BERT model checkpoint |
| `output/binary_model/final_model/config.json` | BERT model configuration |
| `output/binary_model/final_model/model.safetensors` | Model weights (DistilBERT binary classifier) |
| `output/binary_model/final_model/tokenizer_config.json` | Tokenizer configuration |
| `output/binary_model/final_model/tokenizer.json` | Tokenizer vocabulary |
| `governance_config.json` | Updated thresholds (Youden's J, confidence gates) |


### 2. Data Ingestion (Reddit API)

**Sources**: 21 mental health communities
- mentalhealth, depression, SuicideWatch, anxiety, stress, offmychest, lonely, BPD, PTSD, socialanxiety, bipolar, addiction, traumatoolbox, CPTSD, selfharm, OCD, EatingDisorders, MentalHealthSupport, schizophrenia, insomnia, panicattacks, ADHD

**Volume**: 50 posts per subreddit (~1,050 real posts per execution)  
**Filtering**: Removes deleted, archived posts; keeps title + selftext (max 2000 chars)

**Implementation** (`get_reddit()` in `etl_pipeline_new.py`):
```python
# Authenticate with Reddit API
reddit = authenticate_reddit(client_id, client_secret)

# Fetch 50 posts from each of 21 subreddits
posts = []
for subreddit in SUBREDDIT_LIST:
    posts.extend(fetch_recent_posts(reddit, subreddit, limit=50))

# Return as DataFrame with columns: id, created_utc, subreddit, text, url
return pd.DataFrame(posts)
```

### 3. Text Processing & BERT Inference

**Cleaning**:
- Remove URLs, newlines, special characters
- Expand contractions (e.g., "don't" → "do not")
- Tokenize to max 256 tokens

**BERT Inference**:
- Model: Fine-tuned distilBERT for binary classification
- Output: P(suicidal_ideation) ∈ [0, 1]
- Calibration: Temperature scaling (T=10.0)
- Batch processing: GPU-optimized batches of 16

```python
# Load pre-trained DistilBERT binary classifier
model = load_model("output/binary_model/final_model")
scaler = TemperatureScaler(temperature=10.0)

# Process posts in batches of 16
for batch in batch_iterator(posts, batch_size=16):
    raw_predictions = model.predict(batch)
    calibrated_probs = scaler.calibrate(raw_predictions)
    # Output: [P(non_suicidal), P(suicidal)] for each post
```

### 4. Geolocation & County Mapping

**Location Extraction**:
- spaCy NER extracts location entities (GPE type) from post text
- Geocoding maps location to coordinates via Nominatim (OpenStreetMap)
- **Caching**: Same location queried multiple times → cached to avoid re-requests
- **Rate Limiting**: 2-second inter-request delay to respect Nominatim rate limits (~1 req/sec)
- **Fallback**: Posts without location mentioned are geolocated as `(None, None, None)` and excluded from county aggregation


```python
# Geolocation with caching and rate limiting
geocoding_cache = {}
for text in posts:
    # Extract location entities from post text
    locations = extract_entities_spacy(text, entity_type="GPE")
    
    # Query cache first, then Nominatim API
    for location in locations:
        if location in geocoding_cache:
            coords = geocoding_cache[location]
        else:
            coords = geocode_nominatim(location)  # API call
            geocoding_cache[location] = coords
            sleep(2)  # Rate limiting: ~1 req/sec
        
        if coords:
            break
    
    geo_results.append((location, coords))
```

### 5. Crisis Signal Design (Post-Level Classification)

**Binary Classification**:
- BERT model predicts: P(Suicidal Ideation) ∈ [0, 1]
- Output: risk_tier ∈ {0, 1}
  - **Tier 0**: Non-Suicidal (risk_tier=0)
  - **Tier 1**: Suicidal ideation detected (risk_tier=1)

**Confidence Calculation**:
```python
# Confidence is the maximum probability across [P(non_suicidal), P(suicidal)]
confidences = maximum_probability(bert_probs)  # [0, 1] per post
```

**Auxiliary Signals**:
- TextBlob sentiment polarity: -1 (negative) to +1 (positive)
- Used for context but not primary classification driver

**Human Review Gate**:
```python
review_threshold = 0.60

for post in predictions:
    if post.confidence < review_threshold:
        flag_for_human_review(post)
        # Post added to human_review.csv regardless of predicted tier
    else:
        add_to_archive(post)  # Archive by tier (suicidal vs non-suicidal)
```

**Risk Factors Explanation** (for audit trail):
```python
def create_risk_explanation(post):
    factors = []
    if post.risk_tier == 1:
        factors.append(f"Suicidal pattern ({post.prob_suicidal}%)")
    if post.sentiment < -0.3:
        factors.append("Strong negative sentiment")
    if post.confidence < 0.7:
        factors.append("Moderate model confidence")
    return " | ".join(factors) or "Non-suicidal"
```

**Note**: This is post-level classification. County-level aggregation (72-hour windowing for Deliverable 1) can be computed downstream from the `suicidal_detection.csv` output for reporting purposes.

### 6. Governance Controls (Deliverable 2)

**Classification Tiers** (Post-Level):
- **Tier 0 (Non-Suicidal)**: risk_tier=0 → `non_suicidal_detection.csv`
- **Tier 1 (Suicidal Ideation)**: risk_tier=1 → `suicidal_detection.csv`

**Human Review Escalation**:
All predictions with **confidence < 0.60** are flagged to `human_review.csv`:
- Low-confidence Tier-0 predictions ("might be suicidal but model unsure")
- Low-confidence Tier-1 predictions ("model says suicidal but low confidence")
- These require analyst verification before any action

**Safeguards Applied**:

| Safeguard | Purpose | Implementation | Location |
|-----------|---------|---|----------|
| **Bot Detection** | Flag coordinated/spam activity | K-S statistical test on score distribution | `drift_detection.py` → `DriftDetector.detect_drift()` |
| **Media Filtering** | Remove news-driven sentiment spikes | Detect uniform sentiment across locations | `governance.py` → risk tier assignment logic |
| **Rural Equity** | Prevent underrepresentation | Min 20 posts threshold + confidence penalty | `governance.py` → `apply_cumulative_thresholds()` |
| **Drift Detection** | Monitor model performance degradation | K-S test: baseline vs. new predictions | `drift_detection.py` |

### Safeguard 1: Bot Detection & Coordinated Activity

**Threat**: Coordinated social media accounts amplify false signals.

**Detection Mechanism** (`drift_detection.py`):
- Kolmogorov-Smirnov test (K-S test) compares current vs. baseline distribution
- If p-value < 0.05: Statistical shift detected
- Flagged to analyst as "possible coordinated activity"

**Implementation** (pseudocode):
```python
detector = DriftDetector(baseline_scores=historical_data)
has_drift = detector.detect_drift(current_scores)

if has_drift:
    flag_escalation("statistically_anomalous")
    # Alerts analyst: suspicious pattern
```

**Outcome**: 
- Escalation flagged as "statistically anomalous"
- Analysts apply higher skepticism before action
- Reduces false escalations from coordinated bots

---

### Safeguard 2: Media Event Filtering

**Threat**: News coverage of suicides or celebrity deaths triggers spikes unrelated to community crisis.

**Detection** (`governance.py` → `_detect_media_spike()`):
- Z-score test on post volume: If current_count > 2.5σ above baseline, flag as media spike
- Maintains rolling baseline per county (mean + std) updated each window
- Triggers when post count is statistically anomalous

**Implementation** (pseudocode):
```python
def _detect_media_spike(posts, county):
    current_count = len(posts)
    baseline = region_baseline[county]  # {mean, std, count}
    
    # Compute z-score
    z_score = (current_count - baseline['mean']) / baseline['std']
    
    # Update rolling baseline (Welford's algorithm)
    baseline['mean'] += (current_count - baseline['mean']) / (baseline['count'] + 1)
    baseline['std'] = update_variance(baseline, current_count)
    
    # Flag if anomalous
    if z_score > 2.5:  # p < 0.01
        return True
    return False
```

**Outcome**:
- Flagged posts tagged with `'potential_media_spike'` in decision record
- Escalations marked as `'flagged_for_review'` tier
- Analysts see spike context and determine if action needed

**Note**: Media spikes are NOT ignored. They are escalated with context so analysts can distinguish genuine community crisis from news-driven sentiment.

---

### Safeguard 3: Rural Equity & Minimum Sample Size

**Threat**: Rural counties have fewer posts. Without thresholds, system falsely excludes them.

**Solution**:
```
Standard threshold: 20 posts per county per window
Rural adjustment: For counties <100k population,
    adjusted_threshold = MAX(10, expected_posts_per_window × 0.1)
```

**Rationale**: Don't penalize rural counties for low volume, but maintain statistical rigor.

---

### Safeguard 4: Drift Detection

**Threat**: BERT model performs differently over time (distribution shift, data decay).

**Implementation** (pseudocode):
```python
from drift_detection import DriftDetector

detector = DriftDetector()
has_drift = detector.check_drift(current_predictions)

if has_drift:
    log_alert("Statistical drift detected")
    if consecutive_alerts >= 3:
        trigger_model_recalibration()
```

**Baseline**: Persisted in `drift_baseline.pkl` for comparison  
**Trigger**: Monthly OR after 3 consecutive drift detections  
**Action**: Refit temperature T + recompute Youden's J

---

## **Audit Logging** (per post):

```python
# Each post record saved to CSV includes:
post_record = {
    'id': post_id,                          # Unique Reddit post ID
    'subreddit': subreddit,                 # Source subreddit
    'text': post_text,                      # Full post content
    'status': 'Suicidal' or 'Non-Suicidal', # Classification
    'confidence': float_0_to_1,             # Model confidence [0, 1]
    'prob_suicidal': float,                 # P(Suicidal Ideation)
    'prob_non_suicidal': float,             # P(Non-Suicidal)
    'risk_tier': 0 or 1,                    # Risk classification tier
    'sentiment': float_neg1_to_1,           # TextBlob sentiment[-1, 1]
    'needs_human_review': bool,             # True if confidence < 0.60
    'risk_factors': str,                    # Explanation of decision
    'location_name': str,                   # Extracted location
    'lat': float, 'lon': float,             # Coordinates
    'created_utc': timestamp,               # Post creation time
    'model_version': 'bert_v1'              # Model identifier
}
```

All decisions are **immutable** once persisted to CSV, with full traceability.

---

## Files

| File | Purpose |
|------|---------|
| `etl_pipeline_new.py` | Main orchestration + error handling |
| `bert_inference.py` | Model loading + batch inference |
| `calibration.py` | Temperature scaling (T=10.0) |
| `crisis_scoring.py` | Composite score calculation (Deliverable 1) |
| `geocoding.py` | Location extraction + county mapping |
| `governance.py` | Safeguards + risk tier assignment (Deliverable 2) |
| `continuous_eval.py` | Drift detection + health monitoring |
| `train_bert_binary.py` | Fine-tuning + calibration + Youden's J threshold |
| `drift_detection.py` | Model performance monitoring via K-S statistical testing |

---

## CSV Outputs at [Outputs](output/evaluations)

| File | Purpose | Record Type |
|------|---------|---|
| `suicidal_detection.csv` | Posts classified as Suicidal (risk_tier=1) | All Tier-1+ detections |
| `non_suicidal_detection.csv` | Posts classified as Non-Suicidal (risk_tier=0) | All Tier-0 posts |
| `human_review.csv` | Low-confidence predictions (confidence < 0.60) | Flagged for analyst review |
| `etl_logs.csv` | Pipeline execution history + metrics | 1 row per execution |
| `drift_baseline.pkl` | Baseline state for drift detection (binary pickle) | Updated after each run |

**Data Persistence Model**:
- CSV files **accumulate** – each execution appends new rows
- post_id acts as unique key (post-level deduplication)
- Drift baseline persists across runs for comparative analysis

---

## Monitoring & Metrics

Each execution logs metrics to `output/evaluations/etl_logs.csv`:

| Metric | Tracked In | Purpose |
|--------|------------|---------|
| **run_timestamp** | etl_logs.csv | When execution started (UTC) |
| **status** | etl_logs.csv | success / error with error message |
| **events_processed** | etl_logs.csv | Total posts fetched + processed |
| **events_escalated** | etl_logs.csv | Count flagged as Suicidal (Tier-1+) |
| **geocoding_success** | Log output | Percentage of posts successfully geolocated |
| **human_review_count** | Log output | Low-confidence posts requiring analyst |
| **drift_status** | Log output (drift baseline tracked offline) | K-S test result, baseline vs. new predictions |

**Example etl_logs.csv entry**:
```
run_timestamp,status,events_processed,events_escalated,error_message,duration_seconds
2024-01-15T14:30:00Z,success,1050,73,,287
```

**Drift Baseline** (`drift_baseline.pkl`):
- Binary pickle file updated after each successful run
- Stores baseline prediction distribution
- Used to detect when model behavior changes significantly
- K-S test triggers warning if drift exceeds threshold (0.05)

---

## Configuration

### Environment Variables [Example](.env.example)

Set these before running the pipeline:

```bash
# Required for Reddit API access
export REDDIT_CLIENT_ID="your_reddit_app_id"
export REDDIT_CLIENT_SECRET="your_reddit_app_secret"

# Optional overrides (defaults work fine)
export BERT_MODEL_DIR="output/binary_model/final_model"
export CALIBRATION_ARTIFACT="output/binary_model/calibration_artifacts.json"
export GOVERNANCE_CONFIG="deliverables/2_governance_controls/governance_config.json"
export DRIFT_BASELINE="output/evaluations/drift_baseline.pkl"

# Optional: Skip geolocation if Nominatim rate limits are problematic
export SKIP_GEOCODING="false"
export MAX_GEOCODING_POSTS="100"
```

### Governance Config

Single source of truth loaded from `deliverables/2_governance_controls/governance_config.json`:

```json
{
  "optimal_threshold": 0.7139,        // Youden's J optimal threshold
  "threshold_monitor": 0.50,          // Tier 0 → Tier 1 boundary
  "threshold_escalate": 0.75,         // Tier 1 → Tier 2 boundary
  "confidence_min": 0.60,             // Human review confidence threshold
  "min_sample_threshold": 20,         // Rural equity: min posts per county
  "ema_alpha": 0.2,                   // Exponential moving average smoothing
  "drift_ks_threshold": 0.05          // K-S test significance level
}
```

The pipeline reads this file at startup to configure:
- Escalation tier thresholds
- Confidence gates
- Drift detection sensitivity




