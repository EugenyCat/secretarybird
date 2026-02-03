"""
Market regime classification for OutlierDetection module.

Classifies market regimes based on pre-computed analyzer characteristics
for intelligent method selection in Enhanced Decision Tree.

MATHEMATICAL CORRECTIONS v2.1:
- Fixed trend_strength validation: enforced [0, 1] bound for R²
- Corrected volatility confidence: proper CDF transformation
- Fixed autocorrelation SE: standard Box & Jenkins formula (1/√n)
- Corrected autocorr confidence: two-sided test with proper p-value
- Added persistence_score clamping for numerical safety
- Improved feature semantic clarity (persistence split, efficiency rename)
- Enhanced numerical stability in confidence calculation

Based on Mathematical Validation Report 2025-10-28 (Audit v1.0)
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
from scipy import stats

from pipeline.helpers.utils import validate_required_locals

__version__ = "2.1.0"

# Minimum sample size for stable regime classification (Box & Jenkins, 1976)
MIN_SAMPLE_SIZE_REGIME = 30

# Thresholds for regime classification (mathematically corrected)
# Traditional markets: based on VIX historical data (CBOE, 1990-2024)
VOLATILITY_THRESHOLDS_TRADITIONAL = {
    "low": 0.05,   # 5% annualized (VIX < 12)
    "high": 0.25   # 25% annualized (VIX > 25)
}

# Crypto markets: ~3x adjustment factor (Liu et al., 2020)
VOLATILITY_THRESHOLDS_CRYPTO = {
    "low": 0.15,   # 15% annualized
    "high": 0.50   # 50% annualized
}

# Trend strength R² thresholds (Hamilton, 1994)
TREND_THRESHOLDS = {
    "weak": 0.10,    # 10% variance explained
    "medium": 0.40   # 40% variance explained
}

# Persistence thresholds (based on Hurst exponent, Peters, 1994)
PERSISTENCE_THRESHOLDS = {
    "low": 0.45,   # H < 0.45 → mean-reverting
    "high": 0.55   # H > 0.55 → trending
}

# Autocorr thresholds (preserved for backward compatibility)
AUTOCORR_THRESHOLDS = {"low": 0.5, "high": 0.9}


def _get_volatility_thresholds(instrument_type: Optional[str] = None) -> Dict[str, float]:
    """
    Select appropriate volatility thresholds based on instrument type.

    Args:
        instrument_type: "crypto"/"cryptocurrency" or None for traditional

    Returns:
        Dict with "low" and "high" volatility thresholds

    References:
        - CBOE VIX methodology for traditional markets
        - Liu, Y. et al. (2020) "Risks and Returns of Cryptocurrency"
    """
    if instrument_type and instrument_type.lower() in ["crypto", "cryptocurrency"]:
        return VOLATILITY_THRESHOLDS_CRYPTO
    return VOLATILITY_THRESHOLDS_TRADITIONAL


def _validate_input_ranges(
    volatility: float, trend_strength: float, autocorr: float, noise_level: float
) -> None:
    """
    Validate mathematical ranges according to time series theory.

    Args:
        volatility: Volatility [0, +∞)
        trend_strength: Trend strength (R²) [0, 1]
        autocorr: Autocorrelation [-1, 1]
        noise_level: Noise level [0, 1]

    Raises:
        ValueError: For invalid mathematical ranges
    """
    if volatility < 0:
        raise ValueError(f"Volatility must be non-negative, got {volatility}")
    if not (0 <= trend_strength <= 1):
        raise ValueError(
            f"Trend strength (R²) must be in [0, 1], got {trend_strength}. "
            f"Check analyzer implementation - R² by definition cannot exceed 1."
        )
    if not (-1 <= autocorr <= 1):
        raise ValueError(f"Autocorrelation must be in [-1, 1], got {autocorr}")
    if not (0 <= noise_level <= 1):
        raise ValueError(f"Noise level must be in [0, 1], got {noise_level}")


def _handle_edge_cases(
    volatility: float,
    trend_strength: float,
    autocorr: float,
    noise_level: float,
    instrument_type: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Handle mathematical edge cases for numerical stability.

    Args:
        volatility: Time series volatility
        trend_strength: Trend strength
        autocorr: Lag-1 autocorrelation
        noise_level: Noise level
        instrument_type: Instrument type for threshold selection

    Returns:
        Dict with result if special case, None if normal processing
    """
    thresholds = _get_volatility_thresholds(instrument_type)

    # Constant time series (volatility ≈ 0)
    if volatility < 1e-10:
        return {
            "market_regime": "constant_series",
            "regime_confidence": 1.0,
            "volatility_regime": "low",
            "trend_regime": "weak",
            "persistence_regime": "low",
            "special_case": True,
        }

    # Perfect autocorrelation (|autocorr| ≈ 1)
    if abs(autocorr) > 0.999:
        regime = "perfect_persistence" if autocorr > 0 else "perfect_reversion"
        return {
            "market_regime": regime,
            "regime_confidence": 1.0,
            "volatility_regime": _classify_volatility_regime(volatility, thresholds),
            "trend_regime": _classify_trend_regime(trend_strength),
            "persistence_regime": "high" if autocorr > 0 else "low",
            "special_case": True,
        }

    # Perfect signal (noise_level ≈ 0)
    if noise_level < 1e-6:
        return {
            "market_regime": "perfect_signal",
            "regime_confidence": 1.0,
            "volatility_regime": _classify_volatility_regime(volatility, thresholds),
            "trend_regime": _classify_trend_regime(trend_strength),
            "persistence_regime": _classify_persistence_regime(autocorr, trend_strength),
            "special_case": True,
        }

    return None  # No special case, continue normal processing


def classify_market_regime(analyzer_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify market regime based on pre-computed analyzer data.

    MATHEMATICAL CORRECTIONS (v2.1):
    - Enforced R² ∈ [0,1] validation
    - Fixed confidence calculation with proper statistical tests
    - Corrected standard error formulas (Box & Jenkins)
    - Improved numerical stability throughout

    Args:
        analyzer_data: Data from context["currentProperties"]["analyzer"]

    Returns:
        Dict with regime classification:
        {
            "market_regime": "volatile_ranging",    # trending/ranging/volatile + combinations
            "regime_confidence": 0.85,              # Classification confidence
            "volatility_regime": "high",            # low/medium/high
            "trend_regime": "weak",                 # strong/medium/weak
            "persistence_regime": "high"            # momentum persistence level
        }

    References:
        - Box & Jenkins (1976) "Time Series Analysis"
        - Fisher (1915) "Frequency Distribution of Correlation Coefficient"
        - Hamilton (1994) "Time Series Analysis"
    """
    try:
        validate_required_locals(
            ["volatility", "estimated_trend_strength", "lag1_autocorrelation"],
            analyzer_data,
        )

        # Extract key characteristics
        volatility = analyzer_data["volatility"]
        trend_strength = analyzer_data["estimated_trend_strength"]
        autocorr = analyzer_data["lag1_autocorrelation"]
        noise_level = analyzer_data.get("noise_level", 0.0)
        is_stationary = analyzer_data.get("is_stationary", 0)
        sample_size = analyzer_data.get("length", 100)  # Default 100 if not provided
        instrument_type = analyzer_data.get("instrument_type", None)

        # Sample size validation (Box & Jenkins, 1976)
        if sample_size < MIN_SAMPLE_SIZE_REGIME:
            logging.warning(
                f"RegimeClassifier - Insufficient data: {sample_size} < {MIN_SAMPLE_SIZE_REGIME}"
            )
            return {
                "status": "error",
                "message": f"Insufficient data for regime classification: {sample_size} < {MIN_SAMPLE_SIZE_REGIME}"
            }

        # Validate mathematical ranges
        _validate_input_ranges(volatility, trend_strength, autocorr, noise_level)

        # Handle edge cases
        edge_case_result = _handle_edge_cases(
            volatility, trend_strength, autocorr, noise_level, instrument_type
        )
        if edge_case_result:
            logging.info(
                f"RegimeClassifier - Edge case: {edge_case_result['market_regime']}"
            )
            return {"status": "success", "result": edge_case_result}

        # Get appropriate thresholds
        vol_thresholds = _get_volatility_thresholds(instrument_type)

        # Classify by components
        volatility_regime = _classify_volatility_regime(volatility, vol_thresholds)
        trend_regime = _classify_trend_regime(trend_strength)
        persistence_regime = _classify_persistence_regime(autocorr, trend_strength)

        # Combined market regime classification
        market_regime = _classify_market_regime(
            volatility_regime, trend_regime, persistence_regime, is_stationary
        )

        # Calculate classification confidence (CORRECTED v2.1)
        confidence = _calculate_regime_confidence(
            volatility=volatility,
            trend_strength=trend_strength,
            autocorr=autocorr,
            noise_level=noise_level,
            sample_size=sample_size,
            vol_thresholds=vol_thresholds
        )

        result = {
            "market_regime": market_regime,
            "regime_confidence": confidence,
            "volatility_regime": volatility_regime,
            "trend_regime": trend_regime,
            "persistence_regime": persistence_regime
        }

        logging.info(
            f"RegimeClassifier - {market_regime} (confidence: {confidence:.2f}, n={sample_size})"
        )

        return {"status": "success", "result": result}

    except Exception as e:
        logging.error(f"RegimeClassifier - Error: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to classify market regime: {str(e)}",
        }


def get_regime_thresholds(regime_type: str) -> Dict[str, float]:
    """
    Dynamic thresholds for various regimes.

    Args:
        regime_type: Regime type ("volatile_trending", "stable_ranging", etc.)

    Returns:
        Dict with dynamic thresholds for OutlierDetection
    """
    validate_required_locals(["regime_type"], locals())

    # Base thresholds for OutlierDetection
    base_thresholds = {
        "outlier_threshold_multiplier": 1.0,
        "contamination_factor": 1.0,
        "isolation_samples": 1.0,
        "lof_neighbors": 1.0,
    }

    # Adapt thresholds to regime
    if "volatile" in regime_type:
        base_thresholds.update(
            {
                "outlier_threshold_multiplier": 1.3,  # More tolerant thresholds
                "contamination_factor": 1.5,
                "isolation_samples": 0.8,
                "lof_neighbors": 1.2,
            }
        )
    elif "trending" in regime_type:
        base_thresholds.update(
            {
                "outlier_threshold_multiplier": 1.1,
                "contamination_factor": 0.8,
                "isolation_samples": 1.2,
                "lof_neighbors": 0.9,
            }
        )
    elif "ranging" in regime_type:
        base_thresholds.update(
            {
                "outlier_threshold_multiplier": 0.9,
                "contamination_factor": 0.7,
                "isolation_samples": 1.1,
                "lof_neighbors": 1.0,
            }
        )

    return base_thresholds


def calculate_regime_features(analyzer_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Additional regime characteristics for Enhanced Decision Tree.

    Args:
        analyzer_data: Data from analyzer module

    Returns:
        Dict with additional features for method selection
    """
    validate_required_locals(
        ["volatility", "estimated_trend_strength", "lag1_autocorrelation"],
        analyzer_data,
    )

    volatility = analyzer_data["volatility"]
    trend_strength = analyzer_data["estimated_trend_strength"]
    autocorr = analyzer_data["lag1_autocorrelation"]
    noise_level = analyzer_data.get("noise_level", 0.0)
    outlier_ratio = analyzer_data.get("outlier_ratio", 0.0)

    # Numerical stability: clamp noise_level to prevent division issues
    noise_level_clamped = np.clip(noise_level, 0.001, 0.999)

    # Additional features for ML methods
    features = {
        "trend_volatility_ratio": trend_strength / max(volatility, 0.001),
        "signal_noise_ratio": np.clip(
            (1 - noise_level_clamped) / noise_level_clamped,
            0,
            100  # Cap at 100 (40 dB)
        ),
        # Separated persistence features for clarity (v2.1)
        "trending_persistence": max(autocorr, 0) * trend_strength,
        "reverting_tendency": max(-autocorr, 0) * (1 - trend_strength),
        "regime_stability": 1.0 - (volatility * noise_level),
        "outlier_normalized": min(outlier_ratio / 0.1, 1.0),
        # Renamed for semantic clarity: weak-form EMH only (v2.1)
        "weak_form_efficiency": 1.0 - abs(autocorr),
    }

    # Catch22 derived features (if available)
    if "c22_dfa" in analyzer_data:
        # DFA (Detrended Fluctuation Analysis) - canonical persistence measure
        features["fractal_dimension"] = analyzer_data["c22_dfa"]
        features["hurst_proxy"] = analyzer_data["c22_dfa"]  # DFA ≈ Hurst exponent

    if "c22_entropy_pairs" in analyzer_data:
        features["complexity_measure"] = analyzer_data["c22_entropy_pairs"]

    if "c22_ami_timescale" in analyzer_data:
        features["memory_timescale"] = min(
            analyzer_data["c22_ami_timescale"] / 100.0, 1.0
        )

    return features


# ========== HELPER FUNCTIONS ==========


def _classify_volatility_regime(
    volatility: float,
    thresholds: Optional[Dict[str, float]] = None
) -> str:
    """
    Classify volatility with adaptive thresholds.

    Args:
        volatility: Volatility (normalized or annualized)
        thresholds: Thresholds {"low": X, "high": Y}, if None - traditional used

    Returns:
        "low", "medium", or "high"
    """
    if thresholds is None:
        thresholds = VOLATILITY_THRESHOLDS_TRADITIONAL

    if volatility < thresholds["low"]:
        return "low"
    elif volatility > thresholds["high"]:
        return "high"
    else:
        return "medium"


def _classify_trend_regime(trend_strength: float) -> str:
    """
    Classify trend strength based on R² (Hamilton, 1994).

    Args:
        trend_strength: R² from linear regression [0, 1]

    Returns:
        "weak", "medium", or "strong"

    Note:
        R² = 0.10 → 10% variance explained (weak)
        R² = 0.40 → 40% variance explained (strong)
    """
    if trend_strength < TREND_THRESHOLDS["weak"]:
        return "weak"
    elif trend_strength > TREND_THRESHOLDS["medium"]:
        return "strong"
    else:
        return "medium"


def _classify_persistence_regime(autocorr: float, trend_strength: float) -> str:
    """
    Classify persistence (momentum persistence).

    IMPROVED v2.1: Added clamping for numerical safety.

    Args:
        autocorr: Lag-1 autocorrelation [-1, 1]
        trend_strength: R² trend strength [0, 1]

    Returns:
        "low", "medium", or "high"

    References:
        - Peters, E. (1994) "Fractal Market Analysis"
        - Peng, C.K. et al. (1994) "Detrended Fluctuation Analysis"
    """
    # Ensure trend_strength is in valid range for safety
    trend_strength_safe = np.clip(trend_strength, 0, 1)

    # Improved formula: combines autocorr with trend for better persistence measure
    # Autocorr dominant (weight 0.7), trend reinforcement (weight 0.3)
    persistence_score = 0.7 * autocorr + 0.3 * trend_strength_safe

    if persistence_score < PERSISTENCE_THRESHOLDS["low"]:
        return "low"
    elif persistence_score > PERSISTENCE_THRESHOLDS["high"]:
        return "high"
    else:
        return "medium"


def _classify_market_regime(
    volatility_regime: str,
    trend_regime: str,
    persistence_regime: str,
    is_stationary: int,
) -> str:
    """
    Combined market regime classification.

    Args:
        volatility_regime: "low", "medium", "high"
        trend_regime: "weak", "medium", "strong"
        persistence_regime: "low", "medium", "high"
        is_stationary: 0 or 1

    Returns:
        Classified regime (str)
    """
    # Main classification patterns
    if volatility_regime == "high":
        if trend_regime in ["strong", "medium"]:
            return "volatile_trending"
        else:
            return "volatile_ranging"

    elif trend_regime == "strong":
        if persistence_regime == "high":
            return "strong_trending"
        else:
            return "weak_trending"

    elif trend_regime in ["weak", "medium"] and persistence_regime == "low":
        if is_stationary:
            return "stable_ranging"
        else:
            return "ranging"

    else:
        # Default classification
        if volatility_regime == "low" and trend_regime == "weak":
            return "stable_ranging"
        else:
            return "mixed_regime"


def _calculate_regime_confidence(
    volatility: float,
    trend_strength: float,
    autocorr: float,
    noise_level: float,
    sample_size: int,
    vol_thresholds: Dict[str, float]
) -> float:
    """
    Calculate regime classification confidence (CORRECTED v2.1).

    MATHEMATICAL CORRECTIONS v2.1:
    - Fixed volatility SE: standard formula (1/√(2n))
    - Fixed volatility confidence: symmetric CDF transformation
    - Fixed autocorr SE: Box & Jenkins standard (1/√n)
    - Fixed autocorr confidence: proper two-sided test
    - Enhanced numerical stability with epsilon protection

    Args:
        volatility: Volatility
        trend_strength: R² trend strength
        autocorr: Lag-1 autocorrelation
        noise_level: Noise level [0, 1]
        sample_size: Sample size (n)
        vol_thresholds: Volatility thresholds {"low": X, "high": Y}

    Returns:
        float: Confidence score [0, 1]

    References:
        - Fisher, R.A. (1915) "Z-transformation for correlation"
        - Box & Jenkins (1976) "Time Series Analysis", Ch. 2.1.5
        - Kendall & Stuart (1977) "The Advanced Theory of Statistics"
    """
    # Standard errors with numerical stability
    epsilon = 1e-6

    # 1. Volatility classification confidence
    # Standard error: σ/√(2n) for sample standard deviation
    volatility_se = volatility / max(np.sqrt(2 * sample_size), epsilon)

    vol_distances = [
        abs(volatility - vol_thresholds["low"]),
        abs(volatility - vol_thresholds["high"])
    ]
    vol_z_score = min(vol_distances) / max(volatility_se, epsilon)

    # Symmetric transformation: CDF(z) ∈ [0.5, 1] → [0, 1]
    vol_confidence = 2 * stats.norm.cdf(vol_z_score) - 1
    vol_confidence = np.clip(vol_confidence, 0, 1)

    # 2. Trend classification confidence (Chi-squared test for R²)
    # Under H0 (no trend), R²*n follows chi-squared distribution with df=1
    if sample_size > 10:
        trend_chi2_stat = trend_strength * sample_size
        trend_p_value = 1 - stats.chi2.cdf(trend_chi2_stat, df=1)
        trend_confidence = 1 - trend_p_value
    else:
        # Too small sample, use simple threshold distance
        trend_distances = [
            abs(trend_strength - TREND_THRESHOLDS["weak"]),
            abs(trend_strength - TREND_THRESHOLDS["medium"])
        ]
        trend_confidence = min(max(trend_distances) / max(0.1, epsilon), 1.0)

    # 3. Autocorrelation confidence (Box & Jenkins standard)
    # Standard error for lag-1 autocorrelation: 1/√n under white noise H0
    if sample_size > 3:
        autocorr_se = 1.0 / np.sqrt(sample_size)
        autocorr_z_score = abs(autocorr) / autocorr_se

        # Two-sided test: how significant is autocorr vs zero?
        autocorr_p_value = 2 * stats.norm.sf(autocorr_z_score)
        autocorr_confidence = 1 - autocorr_p_value
        autocorr_confidence = np.clip(autocorr_confidence, 0, 1)
    else:
        # Too small sample, use absolute value as proxy
        autocorr_confidence = abs(autocorr)

    # 4. Signal quality confidence
    signal_confidence = 1.0 - noise_level

    # Weighted combination (theory-driven weights)
    # Volatility and trend most important (70% combined)
    weights = np.array([0.35, 0.35, 0.20, 0.10])
    confidences = np.array([
        vol_confidence,
        trend_confidence,
        autocorr_confidence,
        signal_confidence
    ])

    # Clip to valid range [0, 1]
    confidences = np.clip(confidences, 0, 1)

    # Weighted average
    combined_confidence = np.dot(weights, confidences)

    # Apply sample size penalty for small samples (n < 50)
    if sample_size < 50:
        sample_penalty = sample_size / 50.0
        combined_confidence *= sample_penalty

    return float(np.clip(combined_confidence, 0, 1))