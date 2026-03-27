# AI4MH: AI-Powered Behavioral Analysis for Suicide Prevention
## Governance-Ready Crisis Signal Detection with Human-in-the-Loop Oversight

**Google Summer of Code 2026 – GSoC Task Submission**  
**Organization**: Institute for Social Science Research (ISSR), The University of Alabama  
**Mentor**: David M. White, MPH, MPA

---

## Overview

This system monitors aggregated public sentiment from Reddit for indicators of suicidal ideation across geographic regions. When crisis signals are detected, a multi-stage governance funnel determines whether escalation to human experts is warranted.

**Three Required GSoC Deliverables**:

| # | Deliverable | Purpose | 
|---|-------------|---------|
| **1** | **Crisis Signal Design** | BERT classification + confidence calibration + county-level crisis scoring | 
| **2** | **Governance & Risk Controls** | Escalation logic, safeguards (bot/media/rural), audit logging | 
| **3** | **Governance Reflection** | Primary deployment risk + critical safeguard identification | 

**For detailed implementation**, see [README](FULL_IMPLEMENTATION.md)

---

## System Architecture

```
SIGNAL DETECTION LAYER (Deliverable 1)
├─ BERT Binary Classifier: Detects suicidal ideation in text
├─ Temperature Scaling: Calibrates confidence scores
├─ County-Level Aggregation: Combines individual post scores
├─ Crisis Score: Composite [0,1] with confidence gates

         ↓

GOVERNANCE LAYER (Deliverable 2)
├─ Escalation Tiers: Tier 0 (no action) → Tier 1 (monitor) → Tier 2 (crisis)
├─ Safeguards: Bot detection, Media filtering, Rural equity checks
├─ Drift Detection: Monitors model performance over time
└─ Audit Trail: Logs all decisions with reasoning

         ↓

HUMAN REVIEW LAYER (Deliverable 3)
├─ Analyst Queue: Lists escalations with full context
├─ Human Override: Analyst can reject or approve system decisions
└─ Feedback Loop: Analyst decisions feed back to model improvements
```
---

## Key Implementation Details

### Signal Detection (Deliverable 1: BERT Classification)

**BERT Binary Classifier**
- Detects presence/absence of suicidal ideation in post text
- Input: Reddit post (title + content, max 256 tokens)
- Output: P(suicidal ideation) ∈ [0,1]
- Model: Fine-tuned BERT-base on labeled mental health post corpus
- Location: [deliverables/1_crisis_signal_design/](deliverables/1_crisis_signal_design/)

**Confidence Calibration** 
- Raw model outputs are often miscalibrated (over/underconfident)
- Temperature scaling softens probabilities to match true accuracy
- Achieved calibration: ECE=0.4294 (15.3% improvement)
- Optimal threshold: Youden's J = 0.7139 (from 1,597 validation samples)

**County-Level Aggregation**
- Posts geocoded by extracting location mentions (spaCy NER)
- Scores aggregated by county over 72-hour windows
- Exponential moving average (α=0.2) smooths day-to-day noise
- Sample size gates: N < 20 posts reduces confidence by 50% (rural equity)

### Escalation Logic (Deliverable 2: Governance Engine)

**Three-Tier System**:
- **Tier 0 (Normal)**: Composite crisis score < 0.50 → No action
- **Tier 1 (Monitor)**: 0.50 ≤ composite score < 0.75 → Analyst queue
- **Tier 2 (Crisis)**: Composite score ≥ 0.75 → Immediate escalation + mandatory analyst review

**Safeguards**:
- **Bot Detection**: K-S statistical test flags anomalous posting patterns
- **Media Filtering**: Detects uniform sentiment (known news event or coordinated activity)
- **Rural Equity**: Population-aware confidence thresholds prevent underrepresentation
- **Drift Detection**: Monitors if model performance degrades over time

### Governance Reflection (Deliverable 3: Responsible AI)

**Primary Risk of Premature Deployment**:
The system could become a mass surveillance tool for oppression rather than care:
- False positives on crisis intervention language ("I'm struggling" vs actual ideation risk)
- Algorithmic bias amplification against certain demographics
- Privacy violations enabling workplace discrimination
- Loss of human judgment and context awareness

**Most Critical Safeguard: Human-in-the-Loop Review**
- ALL Tier 2 escalations require mandatory human analyst review
- Analyst reads full post context (not just a score)
- Analyst can override system decision
- Every decision logged for accountability
- Analyst feedback drives model improvements

This preserves human authority over high-stakes decisions.  

## Visualizations & Analysis

| Figure | Content | Insight |
|--------|---------|---------|
| **Fig 1: Risk Distribution** | Crisis signal distribution across counties + human review escalations | Demonstrates risk stratification and human review coverage |
| **Fig 2: Sentiment Analysis** | Sentiment polarity histogram (suicidal vs non-suicidal) | Shows semantic signal strength for calibration |
| **Fig 3: Confidence Metrics** | BERT confidence boxplot by classification | Validates model calibration quality (ECE=0.4294) |
| **Fig 4: Crisis Signal Design** | Sentiment intensity histogram + governance thresholds | Illustrates core scoring mechanism and tier boundaries |

**View figures**: [output/visualizations/](output/visualizations/)

---

## File Structure

```
|── dataset/                           (Input Reddit + Twitter datasets)
├── deliverables/
│   ├── 1_crisis_signal_design/        (Deliverable 1: Crisis Scoring Framework)
│   ├── 2_governance_controls/         (Deliverables 2 & 3: Governance + Reflection)
│   └── 3_monitoring_integration/      (ETL Pipeline & Integration)
├── output/
│   ├── visualizations/                (Fig 1-4 publication-quality PNG)
│   └── evaluations/                   (CSV exports: human_review, suicidal_detection, etc.)
├── README.md                          (This file - Overview & GSoC alignment)
├── FULL_IMPLEMENTATION.md             (Detailed implementation details with pseudocode)
└── requirements.txt
```

---
## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run ETL Pipeline
```bash
export GOVERNANCE_CONFIG=deliverables/2_governance_controls/governance_config.json
python3 deliverables/3_monitoring_integration/etl_pipeline_new.py
```

### Generate Visualizations
```bash
python3 generate_visualizations.py
```

---

## Key Libraries Used

| Library | Purpose |
|---------|---------|
| **torch** | Deep learning framework for BERT model inference |
| **transformers** | HuggingFace library for fine-tuned BERT binary classifier |
| **pandas** | Data manipulation, CSV processing, DataFrame operations |
| **praw** | Reddit API client for fetching posts from mental health subreddits |
| **spacy** | Named Entity Recognition (NER) for location extraction in posts |
| **geopy** | Nominatim geocoding to convert location entities to coordinates |
| **scikit-learn** | ML metrics (ECE, calibration), model evaluation utilities |
| **numpy** | Numerical computing and array operations |
| **scipy** | Statistical tests (K-S test for bot detection) |
| **textblob** | Sentiment polarity analysis for posts |
| **nltk** | Natural language tokenization and text processing |
| **contractions** | Text normalization (expand contractions like "don't" → "do not") |
| **python-dotenv** | Environment variable management (.env file support) |
| **requests** | HTTP library for API calls (Nominatim, other services) |
| **python-dateutil** | Date/time utilities for temporal windowing |
| **pycountry** | Country/region code validation |

--- 
**Author**
- **Name**: Devi Sri Bandaru
- **Email**: [bandarudevisri.ds@gmail.com](bandarudevisri.ds@gmail.com)
- **LinkedIn**: [https://linkedin.com/in/devisri-bandaru](https://linkedin.com/in/devisri-bandaru)
- **GitHub**: [https://github.com/Devisri-B](https://github.com/Devisri-B)
- **For GSoC 2026**