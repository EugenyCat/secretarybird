"""
Statistical analysis method for time series.

Mathematical improvements (v2.0.1):
- ACF bias correction via statsmodels (Box & Jenkins, 1976)
- Optimized entropy via scipy.stats (Shannon, 1948)
- Robust multimodal detection via scipy.signal (Silverman, 1981)
- Enhanced numerical stability and edge case handling
- CRITICAL FIX v2.0.1: Restored all original result keys for backward compatibility

Version: 2.0.1
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pycatch22
from scipy.signal import find_peaks
from scipy.stats import entropy as scipy_entropy
from scipy.stats import norm, probplot
from statsmodels.tsa.stattools import acf

from pipeline.helpers.utils import (
    calculate_volatility,
    estimate_noise_level,
    estimate_trend_strength,
    validate_required_locals,
)
from pipeline.timeSeriesProcessing.analyzer.methods.baseAnalysisMethod import (
    BaseAnalysisMethod,
)

__version__ = "2.0.1"


class StatisticalMethod(BaseAnalysisMethod):
    """
    Statistical analysis method for time series.

    Includes basic statistics, volatility, trend, noise, and bias-corrected
    autocorrelation analysis.
    """

    DEFAULT_CONFIG = {
        **BaseAnalysisMethod.DEFAULT_CONFIG,
        # Adapted by configAnalyzer:
        # "calculate_advanced": True,
        # "autocorr_max_lag": 50,
        # "min_data_for_advanced": 30,
        # "autocorr_significance_level": 0.05,
        # "entropy_bins": 20,
        # "max_significant_lags": 10,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize statistical analysis method.

        Args:
            config: Configuration with parameters
        """
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged_config)

        validate_required_locals(
            [
                "calculate_advanced",
                "autocorr_max_lag",
                "min_data_for_advanced",
                "autocorr_significance_level",
                "entropy_bins",
                "max_significant_lags",
            ],
            self.config,
        )

    def __str__(self) -> str:
        """String representation for logging."""
        return (
            f"StatisticalMethod(calculate_advanced={self.config['calculate_advanced']}, "
            f"autocorr_max_lag={self.config['autocorr_max_lag']}, "
            f"min_data_for_advanced={self.config['min_data_for_advanced']})"
        )

    def process(
        self, data: pd.Series, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform statistical analysis of time series.

        Args:
            data: Time series for analysis
            context: Context with additional information

        Returns:
            Dict with standard format result
        """
        try:
            # Input validation
            validation = self.validate_input(data)
            if validation["status"] == "error":
                return validation

            # Extract context parameters
            context_params = self.extract_context_parameters(context)

            # Standard logging
            self.log_analysis_start(data, context_params)

            # Prepare clean data
            clean_data = self.prepare_clean_data(data, drop_na=True, min_length=1)

            result = {}

            # 1. Basic statistical characteristics
            basic_stats = self._calculate_basic_stats(clean_data, data)
            result.update(basic_stats)

            # 2. Volatility and trend metrics
            volatility_stats = self._calculate_volatility_metrics(clean_data)
            result.update(volatility_stats)

            # 3. Autocorrelation analysis (bias-corrected)
            autocorr_stats = self._calculate_autocorrelation(clean_data, self.config)
            result.update(autocorr_stats)

            # 4. Advanced statistics (if enabled and sufficient data)
            if (
                self.config["calculate_advanced"]
                and len(clean_data) >= self.config["min_data_for_advanced"]
            ):
                advanced_stats = self._calculate_advanced_stats(clean_data)
                result.update(advanced_stats)

            # 5. Catch22 features (if enabled and sufficient data)
            if (
                self.config["calculate_advanced"]
                and len(clean_data) >= self.config["min_data_for_advanced"]
            ):
                catch22_stats = self._calculate_catch22_features(clean_data)
                result.update(catch22_stats)

            # Create standard response
            response = self.create_success_response(
                result,
                data,
                context_params,
                {
                    "statistical_components": [
                        "basic",
                        "volatility",
                        "autocorrelation",
                        "advanced",
                    ]
                },
            )

            # Standard logging
            self.log_analysis_complete(response)

            return response

        except Exception as e:
            return self.handle_error(
                e, "statistical analysis", {"data_length": len(data)}
            )

    def _calculate_basic_stats(
        self, clean_data: pd.Series, original_data: pd.Series
    ) -> Dict[str, Any]:
        """
        Calculate basic statistical characteristics (optimized).

        Args:
            clean_data: Clean data without NaN
            original_data: Original data with possible NaN

        Returns:
            Dict with basic statistics
        """
        # Empty data check
        if len(clean_data) == 0:
            return {
                "length": len(original_data),
                "missing_values": int(original_data.isnull().sum()),
                "missing_ratio": 1.0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
                "skewness": 0.0,
                "kurtosis": 0.0,
                "q1": 0.0,
                "q3": 0.0,
                "iqr": 0.0,
            }

        # Vectorized main statistics
        basic_stats = {
            "length": int(len(original_data)),
            "missing_values": int(original_data.isnull().sum()),
            "missing_ratio": float(original_data.isnull().sum() / len(original_data)),
            "mean": float(clean_data.mean()),
            "std": float(clean_data.std()),
            "min": float(clean_data.min()),
            "max": float(clean_data.max()),
            "median": float(clean_data.median()),
            "skewness": float(clean_data.skew()),
            "kurtosis": float(clean_data.kurtosis()),
        }

        # Efficient quartile calculation
        quartiles = clean_data.quantile([0.25, 0.75])
        basic_stats["q1"] = float(quartiles.iloc[0])
        basic_stats["q3"] = float(quartiles.iloc[1])
        basic_stats["iqr"] = basic_stats["q3"] - basic_stats["q1"]

        return basic_stats

    def _calculate_volatility_metrics(self, clean_data: pd.Series) -> Dict[str, Any]:
        """
        Calculate volatility metrics (optimized).

        CRITICAL: Preserves all original keys for backward compatibility.

        Args:
            clean_data: Clean data without NaN

        Returns:
            Dict with volatility metrics
        """
        result = {}

        # Use utility functions (already optimized)
        result["volatility"] = float(calculate_volatility(clean_data))
        result["estimated_trend_strength"] = float(estimate_trend_strength(clean_data))
        result["noise_level"] = float(estimate_noise_level(clean_data))

        # Returns calculation (original implementation - REQUIRED for backward compat)
        if len(clean_data) > 1:
            # Vectorized returns calculation
            returns = clean_data.pct_change().dropna()
            if len(returns) > 0:
                result["returns_volatility"] = float(returns.std())
                result["returns_mean"] = float(returns.mean())
                result["returns_count"] = int(len(returns))
            else:
                result["returns_volatility"] = 0.0
                result["returns_mean"] = 0.0
                result["returns_count"] = 0
        else:
            result["returns_volatility"] = 0.0
            result["returns_mean"] = 0.0
            result["returns_count"] = 0

        return result

    def _calculate_autocorrelation(
        self, clean_data: pd.Series, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate autocorrelation with bias correction.

        Mathematical improvement: Uses statsmodels ACF with adjusted=True
        for unbiased estimates (Box & Jenkins, 1976).

        Args:
            clean_data: Clean data without NaN
            params: Configuration parameters

        Returns:
            Dict with autocorrelation characteristics
        """
        result = {}

        # Minimum length check
        if len(clean_data) <= 2:
            return {
                "lag1_autocorrelation": 0.0,
                "significant_lags": [],
                "has_significant_autocorr": False,
                "autocorr_threshold": 0.0,
            }

        # Adaptive lag calculation
        autocorr_max_lag = params["autocorr_max_lag"]
        max_possible_lag = min(autocorr_max_lag, len(clean_data) // 3)

        # Bias-corrected ACF using statsmodels
        acf_values = acf(
            clean_data,
            nlags=max_possible_lag,
            adjusted=True,  # Bias correction
            fft=True,  # Faster for large series
            missing="none",
        )

        # Lag-1 autocorrelation
        result["lag1_autocorrelation"] = float(acf_values[1])

        # Bartlett's formula for significance threshold
        significance_level = params["autocorr_significance_level"]
        threshold = norm.ppf(1 - significance_level / 2) / np.sqrt(len(clean_data))
        result["autocorr_threshold"] = float(threshold)

        # Significant lags detection
        significant_lags = []
        for lag in range(1, len(acf_values)):
            if abs(acf_values[lag]) > threshold:
                significant_lags.append(
                    {
                        "lag": int(lag),
                        "value": float(acf_values[lag]),
                        "abs_value": float(abs(acf_values[lag])),
                    }
                )

        # Sort by absolute value descending and limit
        significant_lags = sorted(
            significant_lags, key=lambda x: x["abs_value"], reverse=True
        )[: params["max_significant_lags"]]

        result["significant_lags"] = significant_lags
        result["has_significant_autocorr"] = len(significant_lags) > 0
        result["significant_lags_count"] = int(len(significant_lags))

        return result

    def _calculate_advanced_stats(self, clean_data: pd.Series) -> Dict[str, Any]:
        """
        Calculate advanced statistical characteristics (optimized).

        Args:
            clean_data: Clean data without NaN

        Returns:
            Dict with advanced statistics
        """
        result = {}

        # Q-Q correlation for normality check
        qq_correlation = float(probplot(clean_data, dist="norm", plot=None)[1][2])
        result["qq_correlation"] = qq_correlation

        # Excess kurtosis (pandas returns excess kurtosis by default)
        kurtosis_val = clean_data.kurtosis()
        result["excess_kurtosis"] = (
            float(kurtosis_val) if not pd.isna(kurtosis_val) else 0.0
        )

        # Potential multimodality check (improved)
        multimodal_check = self._check_multimodal_distribution(clean_data)
        result["potential_multimodal"] = multimodal_check

        # Entropy (randomness measure) - improved
        entropy_val = self._calculate_entropy(clean_data, self.config["entropy_bins"])
        result["entropy"] = entropy_val

        return result

    def _check_multimodal_distribution(self, clean_data: pd.Series) -> bool:
        """
        Robust multimodal detection using prominence-based peak finding.

        Mathematical improvement: Uses scipy.signal.find_peaks with prominence
        threshold to filter noise-induced peaks (Silverman, 1981).

        Args:
            clean_data: Clean data

        Returns:
            True if potentially multimodal
        """
        # Build histogram
        hist, _ = np.histogram(clean_data, bins="auto")

        # Find peaks with prominence threshold
        # Prominence = vertical distance to neighboring valleys
        prominence_factor = self.config.get('multimodal_prominence_factor', 0.1)
        peaks, _ = find_peaks(hist, prominence=hist.max() * prominence_factor)

        # Multimodal if 2+ significant peaks
        return len(peaks) > 1

    def _calculate_entropy(self, clean_data: pd.Series, bins: int) -> float:
        """
        Calculate Shannon entropy of distribution.

        Mathematical improvement: Uses scipy.stats.entropy for canonical
        implementation (Shannon, 1948).

        Formula: H = -Σ p(x) log(p(x))

        Args:
            clean_data: Clean data
            bins: Number of bins for histogram

        Returns:
            Entropy value
        """
        if clean_data.std() < 1e-10:  # Constant data
            return 0.0

        # Build histogram (counts only)
        effective_bins = min(bins, max(2, len(clean_data) // 2))
        hist, _ = np.histogram(clean_data, bins=effective_bins)

        # Remove zero bins
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0

        # scipy.entropy handles normalization automatically
        return float(scipy_entropy(hist, base=np.e))

    def _calculate_catch22_features(self, clean_data: pd.Series) -> Dict[str, Any]:
        """
        Calculate catch22 features for time series analysis.

        Args:
            clean_data: Cleaned time series data without NaN values

        Returns:
            Dict with catch22 features prefixed with 'c22_'
        """
        result = {}

        # Minimum data length check (needs at least 3 points)
        if len(clean_data) < 3:
            logging.warning(
                f"{self} - Insufficient data for catch22: {len(clean_data)} < 3"
            )
            return result

        try:
            # Convert to list as required by pycatch22
            ts_data = clean_data.tolist()

            # Calculate all catch22 features
            catch22_result = pycatch22.catch22_all(ts_data)

            # Extract feature names and values
            feature_names = catch22_result["names"]
            feature_values = catch22_result["values"]

            # Create result dictionary with c22_ prefix
            for name, value in zip(feature_names, feature_values):
                # Convert long names to short names with c22_ prefix
                short_name = self._get_catch22_short_name(name)

                # Improved NaN handling: skip invalid features
                if not pd.isna(value) and np.isfinite(value):
                    result[f"c22_{short_name}"] = float(value)

        except Exception as e:
            logging.error(f"{self} - Error calculating catch22: {str(e)}")
            result = {}

        return result

    def _get_catch22_short_name(self, long_name: str) -> str:
        """
        Convert catch22 long feature names to short names.

        Args:
            long_name: Original catch22 feature name

        Returns:
            Short name for the feature
        """
        name_mapping = {
            "DN_HistogramMode_5": "mode_5",
            "DN_HistogramMode_10": "mode_10",
            "DN_OutlierInclude_p_001_mdrmd": "outlier_timing_pos",
            "DN_OutlierInclude_n_001_mdrmd": "outlier_timing_neg",
            "CO_f1ecac": "acf_timescale",
            "CO_FirstMin_ac": "acf_first_min",
            "SP_Summaries_welch_rect_area_5_1": "low_freq_power",
            "SP_Summaries_welch_rect_centroid": "centroid_freq",
            "FC_LocalSimple_mean3_stderr": "forecast_error",
            "FC_LocalSimple_mean1_tauresrat": "whiten_timescale",
            "MD_hrv_classic_pnn40": "high_fluctuation",
            "SB_BinaryStats_mean_longstretch1": "stretch_high",
            "SB_BinaryStats_diff_longstretch0": "stretch_decreasing",
            "SB_MotifThree_quantile_hh": "entropy_pairs",
            "CO_HistogramAMI_even_2_5": "ami2",
            "CO_trev_1_num": "trev",
            "IN_AutoMutualInfoStats_40_gaussian_fmmi": "ami_timescale",
            "SB_TransitionMatrix_3ac_sumdiagcov": "transition_variance",
            "PD_PeriodicityWang_th001": "periodicity",
            "CO_Embed2_Dist_tau_d_expfit_meandiff": "embedding_dist",
            "SC_FluctAnal_2_rsrangeﬁt_50_1_logi_prop_r1": "rs_range",
            "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1": "dfa",
        }

        return name_mapping.get(long_name, long_name.lower().replace("_", ""))