"""
Governance engine - county-level crisis signals, Youden's threshold, geocoding, escalation.
"""

import numpy as np 
import pandas as pd
from typing import Dict, List, Tuple, Optional, cast, Any
from datetime import datetime, timedelta
import logging
from scipy import stats
import spacy
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import pycountry


logger = logging.getLogger(__name__)


class GovernanceEngine:
    """Aggregates region-level signals, applies escalation logic and safeguards.
    
    Uses ISO-3166 geographic codes for worldwide support:
    - ISO-3166-1: Country codes (e.g., 'US', 'GB', 'CA')
    - ISO-3166-2: State/Province codes (e.g., 'US-AL', 'GB-ENG', 'CA-ON')
    """
    
    def __init__(
        self,
        min_sample_threshold: int = 20,
        ema_alpha: float = 0.2,
        threshold_monitor: float = 0.5,
        threshold_escalate: float = 0.75,
        confidence_min: float = 0.6,
        persistence_windows: int = 2,
        optimal_threshold: Optional[float] = None
    ):
        self.min_sample_threshold = min_sample_threshold
        self.ema_alpha = ema_alpha
        self.threshold_monitor = threshold_monitor
        self.threshold_escalate = threshold_escalate
        self.confidence_min = confidence_min
        self.persistence_windows = persistence_windows
        self.optimal_threshold = optimal_threshold or 0.70  # Data-driven threshold from Youden's J
        
        # Geographic tracking (ISO-3166 codes for worldwide support)
        self.discovered_countries = {}  # {country_name.lower(): iso_3166_1_code}
        self.discovered_regions = {}  # {region_name.lower(): (iso_3166_2_code, country_code, lat, lon)}
        
        # Initialize spaCy NER model for location extraction
        try:
            self.nlp = spacy.load('en_core_web_sm')
            logger.info("spaCy NER model loaded for location extraction")
        except OSError:
            logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Initialize Nominatim geocoder for dynamic location detection
        self.geocoder = Nominatim(user_agent="crisis_monitor_v1", timeout=cast(Any, 10))
        
        # State for tracking previous region scores (for EMA)
        self.region_scores = {}  # {region_code: prev_score}
        self.region_persistence = {}  # {region_code: num_consecutive_elevated}
        
        # Rolling baselines for spike detection: {region_code: {'mean': float, 'std': float, 'count': int}}
        self.region_baseline_stats = {}
        
        logger.info(f"GovernanceEngine initialized with min_sample={min_sample_threshold}, optimal_threshold={self.optimal_threshold}")
    
    def process_window(self, posts_df: pd.DataFrame, neighbor_counties: Optional[Dict[str, List[str]]] = None, baseline_stats: Optional[Dict[str, Dict]] = None) -> pd.DataFrame:
        """Process posts by county and compute escalation decisions."""
        decisions = []
        
        if posts_df.empty:
            logger.warning("Empty posts dataframe")
            return pd.DataFrame()
        
        # Group by county
        for county_fips, group in posts_df.groupby('county_fips'):
            # Convert county_fips to string (groupby key may be Scalar type)
            county_fips_str: str = str(county_fips)
            decision = self._decide_county(
                county_fips_str,
                group,
                neighbor_counties or {},
                baseline_stats or {}
            )
            decisions.append(decision)
        
        return pd.DataFrame(decisions)
    
    def _decide_county(self, county_fips: str, posts: pd.DataFrame, neighbor_counties: Dict[str, List[str]], baseline_stats: Dict) -> Dict:
        """Compute escalation decision for single county in window."""
        result = {
            'county_fips': county_fips,
            'window_start': posts['created_utc'].min(),
            'window_end': posts['created_utc'].max(),
        }
        
        n = len(posts)
        result['sample_count'] = n
        
        # Check minimum sample size
        if n < self.min_sample_threshold:
            result['decision_tier'] = 'monitor_low_volume'
            result['confidence'] = 0.0
            result['gov_flags'] = ['sparse_data']
            result['sample_adequate'] = False
            return result
        
        result['sample_adequate'] = True
        
        # Compute risk metrics
        risk_rate_suicidal = posts['prob_suicidal'].mean()
        result['risk_rate_suicidal'] = float(risk_rate_suicidal)
        result['risk_rate_high_confidence'] = float((posts['prob_suicidal'] > 0.70).mean())
        
        # Sentiment metrics
        mean_sentiment = posts['sentiment'].mean()
        sentiment_std = posts['sentiment'].std()
        result['mean_sentiment'] = float(mean_sentiment)
        result['sentiment_std'] = float(sentiment_std)
        
        # Volume spike vs baseline
        county_baseline = baseline_stats.get(county_fips, {})
        baseline_volume = county_baseline.get('mean_volume', n)
        baseline_std = county_baseline.get('std_volume', 1.0)
        volume_spike_zscore = (n - baseline_volume) / (baseline_std + 1e-6)
        result['volume_spike_zscore'] = float(volume_spike_zscore)
        
        # Geographic clustering
        neighbor_fips = neighbor_counties.get(county_fips, [])
        geo_cluster_score = 0.0
        if neighbor_fips:
            geo_cluster_score = min(len(neighbor_fips), 5) / 5.0
        result['geo_cluster_score'] = float(geo_cluster_score)
        
        # Composite crisis score
        score_raw = (
            0.40 * risk_rate_suicidal +
            0.25 * max(0, -mean_sentiment) +
            0.20 * max(0, volume_spike_zscore/5) +
            0.15 * geo_cluster_score
        )
        score_raw = np.clip(score_raw, 0.0, 1.0)
        result['score_raw'] = float(score_raw)
        
        # EMA smoothing
        prev_score = self.region_scores.get(county_fips, score_raw)
        score_smooth = self.ema_alpha * score_raw + (1 - self.ema_alpha) * prev_score
        self.region_scores[county_fips] = score_smooth
        result['score_smooth'] = float(score_smooth)
        
        # --- 8. Confidence estimate ---
        # Binomial CI for suicidal rate
        p_suicidal = risk_rate_suicidal
        se_suicidal = np.sqrt(p_suicidal * (1 - p_suicidal) / (n + 1))
        ci_width_suicidal = 1.96 * se_suicidal  # 95% CI
        
        # Sentiment SE
        se_sentiment = sentiment_std / np.sqrt(n + 1)
        
        # Composite confidence: inverse of uncertainty
        max_ci_width = max(ci_width_suicidal, se_sentiment)
        confidence = max(0.0, 1.0 - max_ci_width)
        confidence = np.clip(confidence, 0.0, 1.0)
        result['confidence'] = float(confidence)
        result['ci_width_suicidal_rate'] = float(ci_width_suicidal)
        result['se_sentiment'] = float(se_sentiment)
        
        # --- 9. Governance gates ---
        gov_flags = []
        
        # Bot/coordination check (placeholder)
        bot_score = self._detect_bot_activity(posts)
        if bot_score > 0.7:
            gov_flags.append('possible_bot_activity')
            confidence *= 0.5  # Downgrade confidence
        result['bot_score'] = float(bot_score)
        
        # Media spike check - detects sudden volume increases via z-score on post count
        media_flag = self._detect_media_spike(posts, county_fips=county_fips)
        if media_flag:
            gov_flags.append('potential_media_spike')
        result['media_spike_detected'] = media_flag
        
        # Sparse data flag (already checked above)
        if n < self.min_sample_threshold * 2:
            gov_flags.append('low_sample_volume')
        
        result['gov_flags'] = gov_flags
        
        # --- 10. Persistence check ---
        if score_smooth >= self.threshold_escalate:
            prev_persistence = self.region_persistence.get(county_fips, 0)
            self.region_persistence[county_fips] = prev_persistence + 1
        else:
            self.region_persistence[county_fips] = 0
        
        persistence_count = self.region_persistence[county_fips]
        persistence_ok = persistence_count >= self.persistence_windows
        result['persistence_count'] = persistence_count
        result['persistence_ok'] = persistence_ok
        
        # --- 11. Final escalation decision (using Youden's optimal threshold) ---
        if score_smooth < self.threshold_monitor:
            decision_tier = 'no_action'
        elif score_smooth < self.optimal_threshold:  # Data-driven decision boundary
            decision_tier = 'monitor'
        elif confidence < self.confidence_min:
            decision_tier = 'monitor_low_confidence'
        elif not persistence_ok:
            decision_tier = 'monitor_persistence'
        elif gov_flags:
            decision_tier = 'flagged_for_review'
        else:
            decision_tier = 'escalate_human_review'
        
        result['decision_tier'] = decision_tier
        
        # --- 12. Audit fields ---
        result['audit_ts'] = datetime.utcnow().isoformat()
        result['top_subreddits'] = posts['subreddit'].value_counts().head(3).to_dict()
        
        return result
    
    def _detect_bot_activity(self, posts: pd.DataFrame) -> float:
        """Detect bot behavior: duplicates + burst posting."""
        if len(posts) < 5:
            return 0.0
        
        # Check for repeated phrases
        text_duplicates = posts['text'].value_counts()
        duplicate_ratio = (text_duplicates > 1).sum() / len(text_duplicates)
        
        # Check for burst posting (many posts in short timespan)
        time_diffs = posts['created_utc'].sort_values().diff().dt.total_seconds()
        burst_ratio = (time_diffs < 60).sum() / len(time_diffs)
        
        bot_score = 0.4 * duplicate_ratio + 0.6 * burst_ratio
        return min(1.0, bot_score)
    
    def _detect_media_spike(self, posts: pd.DataFrame, county_fips: Optional[str] = None) -> bool:
        """Detect media-driven spike using z-score on post volume."""
        if not county_fips or len(posts) < 10:
            return False
        
        current_count = len(posts)
        
        if county_fips not in self.region_baseline_stats:
            self.region_baseline_stats[county_fips] = {'mean': float(current_count), 'std': 0.0, 'count': 1}
            return False
        
        baseline = self.region_baseline_stats[county_fips]
        z_score = 0.0 if baseline['std'] < 1e-6 else (current_count - baseline['mean']) / baseline['std']
        
        # Update rolling baseline
        n = baseline['count']
        delta = current_count - baseline['mean']
        baseline['mean'] += delta / (n + 1)
        if n > 0:
            delta2 = current_count - baseline['mean']
            new_variance = (n * baseline['std'] ** 2 + delta * delta2) / (n + 1)
            baseline['std'] = np.sqrt(new_variance)
        baseline['count'] += 1
        
        if z_score > 2.5:
            logger.warning(
                f"Media spike detected in {county_fips}: "
                f"current={current_count}, baseline_mean={baseline['mean']:.1f}, "
                f"z_score={z_score:.2f}"
            )
            return True
        
        return False

    
    def extract_locations_from_text(self, text: str) -> List[str]:
        """Extract ALL geographic locations (GPE entities) from text using spaCy NER."""
        if not self.nlp:
            logger.warning("spaCy model not loaded. Location extraction disabled.")
            return []
        
        try:
            # Process full text (spaCy can handle long text efficiently)
            # GPE = Geopolitical Entity (countries, cities, states)
            doc = self.nlp(text)
            locations = list(set([ent.text for ent in doc.ents if ent.label_ == 'GPE']))
            
            if locations:
                logger.debug(f"Extracted {len(locations)} locations from text: {locations}")
            
            return locations
        except Exception as e:
            logger.error(f"Location extraction failed: {e}")
            return []
    
    def geocode_location(self, location_name: str, country: Optional[str] = None) -> Optional[Tuple[float, float]]:
        # Geocode location via Nominatim, learns geographic units from results
        if not self.geocoder:
            return None
        
        try:
            query = location_name
            if country:
                query = f"{location_name}, {country}"
            
            location: Any = self.geocoder.geocode(query)
            if location:
                # Learn geographic units from Nominatim's address data
                self._learn_geographic_unit_from_location(location, location_name)
                # Location object has latitude and longitude attributes
                lat: float = cast(float, location.latitude)
                lon: float = cast(float, location.longitude)
                return (lat, lon)
            else:
                logger.debug(f"Geocoding returned no results for: {query}")
                return None
        except GeocoderTimedOut:
            logger.warning(f"Geocoding timeout for: {location_name}")
            return None
        except Exception as e:
            logger.error(f"Geocoding error for {location_name}: {e}")
            return None
    
    def _learn_geographic_unit_from_location(self, location, location_name: str):
        """Extract ISO-3166 codes from Nominatim result and cache for worldwide use."""
        try:
            address = location.raw.get('address', {})
            
            # Extract ISO-3166-1 country code (worldwide standard)
            country_code = address.get('country_code', '').upper()
            country_name = address.get('country', '')
            
            if country_code:
                # Learn this country mapping
                self.discovered_countries[country_name.lower()] = country_code
                logger.debug(f"Discovered country: {country_name} → {country_code}")
            
            # Extract ISO-3166-2 state/province code (works for all countries)
            state = address.get('state', '')
            if country_code and state:
                # ISO-3166-2 format: COUNTRY-STATE (e.g., 'US-AL', 'GB-ENG', 'CA-ON')
                iso_3166_2 = f"{country_code}-{state[:3].upper()}"
                location_lower = location_name.lower()
                self.discovered_regions[location_lower] = (
                    iso_3166_2,
                    country_code,
                    location.latitude,
                    location.longitude
                )
                logger.debug(f"Discovered region: {location_name} → {iso_3166_2}")
            
        except Exception as e:
            logger.debug(f"Could not learn geographic unit from location: {e}")
    
    def get_geographic_unit(self, location_name: str, country: Optional[str] = None) -> Tuple[str, str]:
        """Map location to ISO-3166 geographic code (worldwide standard).
        
        Returns:
            Tuple of (iso_code, code_type) where:
            - iso_code: ISO-3166-2 code (e.g., 'US-AL', 'GB-ENG') or ISO-3166-1 (e.g., 'US')
            - code_type: 'ISO_3166_2' (region) or 'ISO_3166_1' (country)
        """
        if not location_name:
            return ("UNKNOWN", "UNKNOWN")
        
        location_lower = location_name.lower().strip()
        
        # FIRST: Check if this region has been discovered before
        if location_lower in self.discovered_regions:
            iso_code, country_code, lat, lon = self.discovered_regions[location_lower]
            return (iso_code, "ISO_3166_2")
        
        # SECOND: Check if it's a discovered country
        if location_lower in self.discovered_countries:
            iso_code = self.discovered_countries[location_lower]
            return (iso_code, "ISO_3166_1")
        
        # THIRD: Check if country is explicitly provided
        if country:
            country_lower = country.lower().strip()
            
            # Check discovered first
            if country_lower in self.discovered_countries:
                return (self.discovered_countries[country_lower], "ISO_3166_1")
            
            # Use pycountry for all 195+ countries worldwide
            if pycountry:
                try:
                    # Try exact match first
                    country_obj = pycountry.countries.search_fuzzy(country)[0]
                    iso_code = country_obj.alpha_2
                    self.discovered_countries[country_lower] = iso_code
                    logger.debug(f"Resolved {country} to {iso_code} via pycountry")
                    return (iso_code, "ISO_3166_1")
                except (LookupError, AttributeError):
                    pass  # Fall through if not found
        
        # FOURTH: Infer country from location name
        for country_name, iso_code in self.discovered_countries.items():
            if country_name in location_lower:
                return (iso_code, "ISO_3166_1")
        
        # FIFTH: Last resort - infer country from location name using pycountry
        if pycountry:
            for country_obj in pycountry.countries:
                country_name = country_obj.name.lower()
                if country_name in location_lower or location_lower in country_name:
                    iso_code = country_obj.alpha_2
                    self.discovered_countries[country_name] = iso_code
                    logger.debug(f"Inferred country from location: {country_name} → {iso_code}")
                    return (iso_code, "ISO_3166_1")
        
        logger.debug(f"Could not map location to geographic unit: {location_name}")
        return ("UNKNOWN", "UNKNOWN")



    def get_discovery_stats(self) -> Dict:
        """Get stats on discovered geographic units (for monitoring)."""
        return {
            'discovered_countries': len(self.discovered_countries),
            'country_list': list(self.discovered_countries.keys()),
            'discovered_regions': len(self.discovered_regions),
            'cached_region_scores': len(self.region_scores),
        }


def compute_youden_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    num_thresholds: int = 100
) -> Tuple[float, float, float]:
    # Find optimal threshold maximizing sensitivity + specificity - 1
    scores = np.asarray(scores).flatten()
    labels = np.asarray(labels, dtype=int).flatten()
    
    thresholds = np.linspace(scores.min(), scores.max(), num_thresholds)
    best_j = -np.inf
    best_threshold: float = float(scores.min())  # Initialize to valid float
    best_sens: float = 0.0
    best_spec: float = 0.0
    
    for t in thresholds:
        pred = (scores >= t).astype(int)
        
        # Sensitivity (recall/TPR)
        tp = ((pred == 1) & (labels == 1)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        sens = float(tp / (tp + fn + 1e-6))
        
        # Specificity (TNR)
        tn = ((pred == 0) & (labels == 0)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        spec = float(tn / (tn + fp + 1e-6))
        
        # Youden's J
        j = sens + spec - 1
        
        if j > best_j:
            best_j = j
            best_threshold = float(t)
            best_sens = sens
            best_spec = spec
    
    return best_threshold, best_sens, best_spec


def set_optimal_threshold_from_validation(
    val_scores: np.ndarray,
    val_labels: np.ndarray
) -> float:
    # Compute optimal threshold from validation data (Youden's J)
    threshold, sensitivity, specificity = compute_youden_thresholds(val_scores, val_labels)
    j_statistic = sensitivity + specificity - 1
    logger.info(
        f"Youden's optimal threshold: {threshold:.4f} | "
        f"Sensitivity={sensitivity:.3f}, Specificity={specificity:.3f}, J={j_statistic:.3f}"
    )
    return threshold

