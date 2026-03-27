"""
Generate visualizations from processed ETL pipeline outputs.
Loads data from suicidal_detection.csv and non_suicidal_detection.csv.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import logging

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Styling
PALETTE = {
    "crisis":    "#E63946",
    "safe":      "#2A9D8F",
    "threshold": "#F4A261",
    "neutral":   "#457B9D",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#F8F9FA",
    "axes.edgecolor": "#CCCCCC",
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# Output directories
VIZ_DIR = Path("output/visualizations")
VIZ_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR = Path("output/evaluations")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Load Youden threshold
try:
    import json
    with open('deliverables/2_governance_controls/governance_config.json') as f:
        YOUDEN_THRESHOLD = json.load(f).get('thresholds', {}).get('youden_j', 0.7139)
except:
    YOUDEN_THRESHOLD = 0.7139

logger.info(f"Using Youden's J threshold: {YOUDEN_THRESHOLD:.4f}")


def load_processed_data():
    """Load suicidal and non-suicidal detection data from CSV files."""
    suicidal_file = EVAL_DIR / "suicidal_detection.csv"
    non_suicidal_file = EVAL_DIR / "non_suicidal_detection.csv"
    
    if not suicidal_file.exists() or not non_suicidal_file.exists():
        logger.error("Missing CSV files. Run ETL pipeline first:")
        logger.error(f"  - {suicidal_file}")
        logger.error(f"  - {non_suicidal_file}")
        sys.exit(1)
    
    df_suicidal = pd.read_csv(suicidal_file)
    df_non_suicidal = pd.read_csv(non_suicidal_file)
    
    df = pd.concat([df_suicidal, df_non_suicidal], ignore_index=True)
    logger.info(f"  Loaded {len(df):,} processed predictions")
    logger.info(f"  - Suicidal: {len(df_suicidal):,}")
    logger.info(f"  - Non-suicidal: {len(df_non_suicidal):,}\n")
    
    return df

def visualize_risk_distribution(df):
    """Figure 1: Risk distribution across model predictions with human review escalations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 1 — Model Predictions on Real Reddit Data (Post-Processing)",
                 fontsize=14, fontweight="bold", y=1.01)
    
    # Left: Histogram
    ax = axes[0]
    scores = df['confidence'].values
    bins = np.linspace(0, 1, 41)
    ax.hist(scores, bins=bins, alpha=0.7, color=PALETTE["neutral"], edgecolor="black", lw=0.5)
    
    suicidal_mask = df['status'] == 'Suicidal'
    ctx_colors = [PALETTE["safe"] if not s else PALETTE["crisis"] for s in suicidal_mask]
    
    ax.axvspan(0.0, 0.3, alpha=0.1, color=PALETTE["safe"], label="Low Confidence (< 0.30)")
    ax.axvspan(0.3, YOUDEN_THRESHOLD, alpha=0.1, color=PALETTE["threshold"], 
               label=f"Medium (0.30 - {YOUDEN_THRESHOLD:.2f})")
    ax.axvspan(YOUDEN_THRESHOLD, 1.0, alpha=0.1, color=PALETTE["crisis"], 
               label=f"High Confidence (> {YOUDEN_THRESHOLD:.2f})")
    
    ax.set_xlabel("Model Confidence Score")
    ax.set_ylabel("Number of Posts")
    ax.set_title("Score Distribution")
    ax.legend(frameon=True, framealpha=0.9, loc='upper right')
    ax.grid(axis="y", alpha=0.4)
    
    # Right: Risk stratification pie chart with human review escalations
    ax = axes[1]
    suicidal_count = (suicidal_mask).sum()
    non_suicidal_count = (~suicidal_mask).sum()
    
    # Load human review escalations
    human_review_count = 0
    human_review_file = EVAL_DIR / "human_review.csv"
    if human_review_file.exists():
        try:
            df_human = pd.read_csv(human_review_file)
            human_review_count = len(df_human)
        except:
            pass
    
    # Create pie chart with three categories
    categories = [non_suicidal_count, suicidal_count, human_review_count]
    labels = [f'Non-Suicidal\n{non_suicidal_count:,}\n({100*non_suicidal_count/len(df):.1f}%)',
              f'Suicidal Detected\n{suicidal_count:,}\n({100*suicidal_count/len(df):.1f}%)',
              f'Human Review\n{human_review_count:,}\n({100*human_review_count/(len(df)+human_review_count):.1f}%)']
    wedge_colors = [PALETTE["safe"], PALETTE["crisis"], PALETTE["threshold"]]
    
    ax.pie(categories, labels=labels, colors=wedge_colors, autopct="", startangle=90,
           textprops={"fontsize": 10, "fontweight": "bold"})
    ax.set_title("Risk Classification & Human Review")
    
    plt.tight_layout()
    path = VIZ_DIR / "fig1_risk_distribution.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {path}")


def visualize_sentiment_geography(df):
    """Figure 2: Sentiment analysis."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Figure 2 — Sentiment Analysis of Reddit Posts",
                 fontsize=14, fontweight="bold", y=1.01)
    
    if 'sentiment' in df.columns:
        sentiments = df['sentiment'].dropna()
        ax.hist(sentiments, bins=30, alpha=0.7, color=PALETTE["neutral"], edgecolor="black", lw=0.5)
        ax.axvline(sentiments.mean(), color=PALETTE["crisis"], lw=2.5, ls="--", 
                   label=f"Mean: {sentiments.mean():.2f}")
        ax.axvline(sentiments.median(), color=PALETTE["safe"], lw=2.5, ls=":", 
                   label=f"Median: {sentiments.median():.2f}")
        ax.axvspan(-1, -0.3, alpha=0.1, color=PALETTE["crisis"], label="Negative sentiment")
        ax.axvspan(0.3, 1, alpha=0.1, color=PALETTE["safe"], label="Positive sentiment")
        
        ax.set_xlabel("Sentiment Score")
        ax.set_ylabel("Number of Posts")
        ax.set_title("Sentiment Distribution")
        ax.legend(frameon=True, framealpha=0.9, loc='upper right')
        ax.grid(axis="y", alpha=0.4)
    
    plt.tight_layout()
    path = VIZ_DIR / "fig2_sentiment_geography.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {path}")


def visualize_performance_metrics(df):
    """Figure 3: Confidence distribution by classification."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Figure 3 — Model Confidence Distribution by Classification",
                 fontsize=14, fontweight="bold", y=1.01)
    
    suicidal_mask = df['status'] == 'Suicidal'
    conf_suicidal = df[suicidal_mask]['confidence']
    conf_non_suicidal = df[~suicidal_mask]['confidence']
    
    bp = ax.boxplot([conf_non_suicidal, conf_suicidal], patch_artist=True, widths=0.55,
                     medianprops=dict(color="white", lw=2.5),
                     boxprops=dict(linewidth=1.5),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    
    bp["boxes"][0].set_facecolor(PALETTE["safe"])
    bp["boxes"][1].set_facecolor(PALETTE["crisis"])
    
    ax.set_xticklabels(["Non-Suicidal", "Suicidal"], fontsize=12, fontweight='bold')
    ax.set_ylabel("Confidence Score", fontsize=12)
    ax.set_title("Confidence Score Distribution by Classification")
    ax.grid(axis="y", alpha=0.4)
    
    # Add summary statistics
    stats_text = f"""Non-Suicidal (n={len(conf_non_suicidal)}): μ={conf_non_suicidal.mean():.3f}, σ={conf_non_suicidal.std():.3f}
Suicidal (n={len(conf_suicidal)}): μ={conf_suicidal.mean():.3f}, σ={conf_suicidal.std():.3f}"""
    
    ax.text(0.98, 0.05, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='#E0E0E0', alpha=0.3, pad=0.8),
            fontfamily='monospace')
    
    plt.tight_layout()
    path = VIZ_DIR / "fig3_performance_metrics.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {path}")


def visualize_crisis_signal_design(df):
    """Figure 4: Crisis Signal Design Framework - Sentiment Intensity Component."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Figure 4 — Crisis Signal Design: Sentiment Intensity Component", 
                 fontsize=14, fontweight="bold", y=1.00)
    
    if 'sentiment' in df.columns:
        sentiments = df['sentiment'].dropna()
        crisis_sentiment = sentiments[sentiments < -0.3]
        normal_sentiment = sentiments[sentiments >= -0.3]
        
        # Create histogram with crisis zone highlighted
        ax.hist([normal_sentiment, crisis_sentiment], bins=25, 
                label=[f'Normal Sentiment (n={len(normal_sentiment)})', 
                       f'Crisis Intensity (n={len(crisis_sentiment)})'], 
                color=[PALETTE["safe"], PALETTE["crisis"]], 
                alpha=0.7, edgecolor="black", lw=1)
        
        # Mark crisis threshold
        ax.axvline(-0.3, color=PALETTE["crisis"], lw=3, ls="--", 
                  label="Crisis Threshold (-0.3)")
        ax.axvspan(-1, -0.3, alpha=0.1, color=PALETTE["crisis"])
        
        # Add statistics
        ax.axvline(sentiments.mean(), color=PALETTE["neutral"], lw=2.5, ls=":", 
                  label=f"Overall Mean: {sentiments.mean():.2f}")
        
        ax.set_xlabel("Sentiment Intensity Score", fontsize=12)
        ax.set_ylabel("Number of Posts", fontsize=12)
        ax.set_title("Sentiment Distribution with Crisis Detection Zones")
        ax.legend(frameon=True, framealpha=0.95, loc='upper left', fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        
        # Add framework box
        framework_text = """Crisis Signal Framework:
• Sentiment < -0.3 → Crisis intensity detected
• Min sample size: n ≥ 20 posts
• Confidence: Bernoulli variance estimate
• Output: Sentiment score ∈ [0,1]
• Integration: Weighted with volume & geography"""
        
        ax.text(0.98, 0.35, framework_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.85, pad=1),
               fontfamily='monospace', linespacing=1.8)
    
    plt.tight_layout()
    path = VIZ_DIR / "fig4_crisis_signal_design.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser(
        description=" Visualizations from processed ETL pipeline CSV outputs")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("  Crisis Detection Visualizations — Processed Data Analysis  ")
    print("="*80 + "\n")
    
    # Load processed data from CSV files
    df = load_processed_data()
    
    if df.empty:
        logger.error("No data available. Processed CSV files are empty.")
        sys.exit(1)
    
    # Generate visualizations
    print(" Generating publication-quality figures…\n")
    visualize_risk_distribution(df)
    visualize_sentiment_geography(df)
    visualize_performance_metrics(df)
    visualize_crisis_signal_design(df)
    
    print(f"\n{'='*80}")
    print(f" SUCCESS — Visualizations generated from processed pipeline data")
    print(f"{'='*80}\n")
    
    print(f" Output saved to: {VIZ_DIR.resolve()}\n")
    
    print(f"Summary:")
    suicidal_count = (df['status'] == 'Suicidal').sum()
    print(f"   Posts analyzed: {len(df):,}")
    print(f"   Suicidal detections: {suicidal_count:,} ({100*suicidal_count/len(df):.1f}%)")
    print(f"   Confidence range: [{df['confidence'].min():.3f}, {df['confidence'].max():.3f}]")
    print(f"   Mean confidence: {df['confidence'].mean():.3f}\n")
    print("Visualizations ready for submission\n")


if __name__ == "__main__":
    main()
