"""
Stationarity analysis method for time series.

Implements rigorous stationarity testing through:
- Augmented Dickey-Fuller (ADF) test for unit root
- Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for trend stationarity
- Rolling window coefficient of variation analysis
- Weighted voting consensus from multiple tests

Mathematical references:
- Said & Dickey (1984): "Testing for Unit Roots in AR-MA Models"
- Kwiatkowski et al. (1992): "Testing the null of stationarity"
- Hamilton (1994): "Time Series Analysis", Chapter 17

Version: 2.0.0 (Mathematical validation compliance)
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from pipeline.helpers.utils import validate_required_locals
from pipeline.timeSeriesProcessing.analyzer.methods.baseAnalysisMethod import (
    BaseAnalysisMethod,
)

__version__ = "2.0.0"


class StationarityMethod(BaseAnalysisMethod):
    """
    Stationarity analysis method using ADF, KPSS, and rolling CV tests.

    Mathematical approach:
    - ADF: Tests H0=unit root (non-stationary). p < alpha → stationary
    - KPSS: Tests H0=stationary. p > alpha → stationary
    - Rolling CV: Empirical stability check via coefficient of variation
    - Weighted voting: Robust consensus from multiple independent tests

    Minimum observations (scientifically validated):
    - ADF: 20+ observations (Said & Dickey, 1984)
    - KPSS: 25+ observations (Lütkepohl & Krätzig, 2004)

    Consensus strategy:
    - Weighted voting replaces strict AND logic
    - ADF: 40% weight, KPSS: 40% weight, Rolling: 20% weight
    - Decision threshold: cumulative score ≥ 0.5
    """

    DEFAULT_CONFIG = {
        **BaseAnalysisMethod.DEFAULT_CONFIG,
        "handle_kpss_errors": True,  # Enable by default for robustness
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize stationarity analysis method.

        Args:
            config: Configuration with parameters:
                - adf_alpha: float (significance level, typically 0.05)
                - kpss_alpha: float (significance level, typically 0.05)
                - rolling_threshold: float (CV threshold for stability)
                - window_ratio: float (rolling window as fraction of data)
                - min_window: int (minimum rolling window size)
                - min_adf_observations: int (minimum 20, default from config)
                - min_kpss_observations: int (minimum 25, default from config)
        """
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged_config)

        validate_required_locals(
            [
                "adf_alpha",
                "kpss_alpha",
                "rolling_threshold",
                "window_ratio",
                "min_window",
                "min_adf_observations",
                "min_kpss_observations",
            ],
            self.config,
        )

        # Validate alpha parameters (0 < alpha < 0.5 for statistical tests)
        self._validate_alpha_parameters()

    def _validate_alpha_parameters(self) -> None:
        """Validate statistical significance levels are in valid range."""
        for param in ["adf_alpha", "kpss_alpha"]:
            alpha = self.config[param]
            if not (0 < alpha < 0.5):
                logging.warning(
                    f"{self} - {param}={alpha} outside typical range (0, 0.5). "
                    f"Clamping to [0.001, 0.499]"
                )
                self.config[param] = max(0.001, min(0.499, alpha))

    def __str__(self) -> str:
        """String representation for logging."""
        return (
            f"StationarityMethod(adf_alpha={self.config['adf_alpha']}, "
            f"kpss_alpha={self.config['kpss_alpha']}, "
            f"rolling_threshold={self.config['rolling_threshold']})"
        )

    def process(
        self, data: pd.Series, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute stationarity analysis on time series.

        Mathematical workflow:
        1. Constant series check (early exit if std < 1e-10)
        2. ADF test (unit root)
        3. KPSS test (trend stationarity)
        4. Rolling window CV analysis (empirical stability)
        5. Weighted voting consensus (40% ADF + 40% KPSS + 20% Rolling)

        Args:
            data: Time series for analysis
            context: Optional context with additional information

        Returns:
            Dict with standard format:
            {
                'status': 'success',
                'result': {
                    'is_stationary': int (0/1),
                    'is_stationary_decision': bool,
                    'adf_pvalue': float,
                    'kpss_pvalue': float,
                    'rolling_mean_cv': float,
                    'weighted_score': float,
                    ... (test details)
                },
                'metadata': {...}
            }
        """
        try:
            # Validation
            validation = self.validate_input(data)
            if validation["status"] == "error":
                return validation

            # Extract context parameters
            context_params = self.extract_context_parameters(context)

            # Logging
            self.log_analysis_start(data, context_params)

            # Prepare clean data
            clean_data = self.prepare_clean_data(data, drop_na=True, min_length=3)

            # Check for constant series (early exit)
            if clean_data.std() < 1e-10:
                logging.warning(f"{self} - Constant series detected (std < 1e-10)")
                return self._create_constant_series_response(
                    clean_data, data, context_params
                )

            result = {}

            # 1. ADF test
            adf_result = self._adf_test(clean_data, self.config)
            result.update(adf_result)

            # 2. KPSS test
            kpss_result = self._kpss_test(clean_data, self.config)
            result.update(kpss_result)

            # 3. Rolling analysis
            rolling_result = self._rolling_analysis(clean_data, self.config)
            result.update(rolling_result)

            # 4. Weighted voting consensus
            is_stationary, weighted_score = self._evaluate_stationarity_weighted(
                result, self.config
            )
            result["is_stationary"] = int(is_stationary)
            result["is_stationary_decision"] = is_stationary
            result["weighted_stationarity_score"] = weighted_score

            # Create standard response
            response = self.create_success_response(
                result,
                data,
                context_params,
                {
                    "stationarity_tests": ["adf", "kpss", "rolling_analysis"],
                    "consensus_method": "weighted_voting",
                    "test_weights": {"adf": 0.4, "kpss": 0.4, "rolling": 0.2},
                },
            )

            # Logging
            self.log_analysis_complete(response)

            return response

        except Exception as e:
            return self.handle_error(
                e, "stationarity analysis", {"data_length": len(data)}
            )

    def _create_constant_series_response(
        self,
        clean_data: pd.Series,
        original_data: pd.Series,
        context_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create response for constant time series (trivially stationary)."""
        result = {
            "is_stationary": 1,
            "is_stationary_decision": True,
            "series_type": "constant",
            "adf_pvalue": np.nan,
            "kpss_pvalue": np.nan,
            "stationary_adf": True,
            "stationary_kpss": True,
            "rolling_mean_cv": 0.0,
            "rolling_std_cv": 0.0,
            "weighted_stationarity_score": 1.0,
            "constant_value": float(clean_data.mean()),
        }

        return self.create_success_response(
            result,
            original_data,
            context_params,
            {"stationarity_tests": ["constant_series_check"]},
        )

    def _adf_test(self, data: pd.Series, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Augmented Dickey-Fuller test for unit root.

        Mathematical formula:
        Δy_t = α + βt + γy_{t-1} + δ_1Δy_{t-1} + ... + δ_pΔy_{t-p} + ε_t

        H0: γ = 0 (unit root exists, non-stationary)
        H1: γ < 0 (no unit root, stationary)

        Decision: p-value < alpha → reject H0 → stationary

        Minimum observations: 20 (Said & Dickey, 1984)

        Args:
            data: Clean time series (no NaN)
            params: Config with adf_alpha and min_adf_observations

        Returns:
            Dict with ADF test results
        """
        min_obs = params["min_adf_observations"]

        if len(data) < min_obs:
            logging.warning(
                f"{self} - Insufficient data for ADF: {len(data)} < {min_obs}. "
                f"Test skipped (Said & Dickey, 1984: minimum 20 obs required)"
            )
            return {
                "adf_pvalue": np.nan,
                "adf_statistic": np.nan,
                "stationary_adf": None,  # Unknown, not False
                "adf_observations_used": len(data),
                "adf_status": "insufficient_data",
            }

        # Execute ADF with automatic lag selection (AIC criterion)
        adf_result = adfuller(data, autolag="AIC")
        adf_pvalue = adf_result[1]
        adf_statistic = adf_result[0]

        # Decision: p < alpha → reject H0 (unit root) → stationary
        stationary_adf = adf_pvalue < params["adf_alpha"]

        return {
            "adf_pvalue": float(adf_pvalue),
            "adf_statistic": float(adf_statistic),
            "stationary_adf": stationary_adf,
            "adf_observations_used": len(data),
            "adf_lags_used": int(adf_result[2]),
            "adf_status": "completed",
        }

    def _kpss_test(self, data: pd.Series, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        KPSS test for trend stationarity.

        Mathematical basis:
        Tests whether a time series is stationary around a deterministic trend.

        H0: Series is (trend) stationary
        H1: Series has unit root

        Decision: p-value > alpha → fail to reject H0 → stationary

        Minimum observations: 25 (Lütkepohl & Krätzig, 2004)

        Args:
            data: Clean time series (no NaN)
            params: Config with kpss_alpha and min_kpss_observations

        Returns:
            Dict with KPSS test results
        """
        min_obs = params["min_kpss_observations"]

        if len(data) < min_obs:
            logging.warning(
                f"{self} - Insufficient data for KPSS: {len(data)} < {min_obs}. "
                f"Test skipped (Lütkepohl & Krätzig, 2004: minimum 25 obs required)"
            )
            return {
                "kpss_pvalue": np.nan,
                "kpss_statistic": np.nan,
                "stationary_kpss": None,  # Unknown, not False
                "kpss_observations_used": len(data),
                "kpss_status": "insufficient_data",
            }

        try:
            # Execute KPSS with automatic lag selection
            kpss_result = kpss(data, regression="c", nlags="auto")
            kpss_pvalue = kpss_result[1]
            kpss_statistic = kpss_result[0]

            # Decision: p > alpha → fail to reject H0 (stationarity) → stationary
            stationary_kpss = kpss_pvalue > params["kpss_alpha"]

            return {
                "kpss_pvalue": float(kpss_pvalue),
                "kpss_statistic": float(kpss_statistic),
                "stationary_kpss": stationary_kpss,
                "kpss_observations_used": len(data),
                "kpss_status": "completed",
            }

        except Exception as e:
            if params["handle_kpss_errors"]:
                logging.warning(
                    f"{self} - KPSS test failed: {str(e)}. Test skipped with NaN."
                )
                return {
                    "kpss_pvalue": np.nan,
                    "kpss_statistic": np.nan,
                    "stationary_kpss": None,  # Unknown, not False
                    "kpss_observations_used": len(data),
                    "kpss_error": str(e),
                    "kpss_status": "error",
                }
            else:
                raise

    def _rolling_analysis(
        self, data: pd.Series, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Rolling window coefficient of variation analysis.

        Empirical stability check: stationary series should have stable
        mean and variance over time.

        CV = std / |mean|

        Low CV (<threshold) indicates stability → stationarity indicator.

        Args:
            data: Clean time series (no NaN)
            params: Config with window_ratio, min_window, rolling_threshold

        Returns:
            Dict with rolling analysis results
        """
        # Adaptive window size with consistent logic
        window_from_ratio = int(len(data) * params["window_ratio"])

        # Ensure: min_window ≤ window ≤ len(data) / 3
        window = max(
            params["min_window"], min(window_from_ratio, len(data) // 3)
        )

        # For very short series: don't exceed 1/3 rule
        if window > len(data) // 3:
            window = max(3, len(data) // 3)

        # Numerical stability: require sufficient data for CV computation
        min_periods_cv = max(3, window // 2)

        # Vectorized rolling computation
        rolling_mean = data.rolling(window=window, min_periods=min_periods_cv).mean()
        rolling_std = data.rolling(window=window, min_periods=min_periods_cv).std()

        # Safe coefficient of variation
        mean_cv = self._safe_coefficient_of_variation(rolling_mean)
        std_cv = self._safe_coefficient_of_variation(rolling_std)

        return {
            "rolling_mean_cv": float(mean_cv),
            "rolling_std_cv": float(std_cv),
            "rolling_window_size": int(window),
            "rolling_window_ratio": float(window / len(data)),
            "rolling_min_periods": int(min_periods_cv),
        }

    def _safe_coefficient_of_variation(self, series: pd.Series) -> float:
        """
        Safe computation of coefficient of variation with numerical stability.

        CV = std / |mean|

        Returns np.inf for degenerate cases (zero mean, NaN values).

        Args:
            series: Time series for CV computation

        Returns:
            Coefficient of variation or np.inf on error
        """
        try:
            mean_val = series.mean()
            std_val = series.std()

            if pd.isna(mean_val) or pd.isna(std_val) or abs(mean_val) < 1e-10:
                return float(np.inf)

            return float(std_val / abs(mean_val))

        except (ZeroDivisionError, ValueError):
            return float(np.inf)

    def _evaluate_stationarity_weighted(
        self, results: Dict[str, Any], params: Dict[str, Any]
    ) -> tuple[bool, float]:
        """
        Weighted voting consensus for stationarity decision.

        Mathematical approach:
        - ADF: 40% weight (primary test, unit root)
        - KPSS: 40% weight (primary test, trend stationarity)
        - Rolling CV: 20% weight (secondary, empirical check)

        Decision threshold: score ≥ 0.5 → stationary

        Adaptive weighting:
        - If KPSS N/A: redistribute 40% weight to ADF (60%) and Rolling (40%)
        - If both ADF and KPSS N/A: rely on Rolling only (100%)

        More robust than strict AND logic for financial data with structural breaks.

        Args:
            results: Results from all tests
            params: Config with rolling_threshold

        Returns:
            Tuple of (is_stationary: bool, weighted_score: float)
        """
        # Extract test results (None = test N/A)
        stationary_adf = results.get("stationary_adf")
        stationary_kpss = results.get("stationary_kpss")
        mean_cv = results.get("rolling_mean_cv", np.inf)
        std_cv = results.get("rolling_std_cv", np.inf)

        # Rolling stability check
        rolling_threshold = params["rolling_threshold"]
        rolling_stable = (
            not np.isinf(mean_cv)
            and mean_cv < rolling_threshold
            and not np.isinf(std_cv)
            and std_cv < rolling_threshold
        )

        # Weighted scoring with adaptive weights
        score = 0.0

        # Case 1: Both ADF and KPSS available (standard weighting)
        if stationary_adf is not None and stationary_kpss is not None:
            if stationary_adf:
                score += 0.4
            if stationary_kpss:
                score += 0.4
            if rolling_stable:
                score += 0.2

        # Case 2: KPSS N/A (redistribute to ADF + Rolling)
        elif stationary_adf is not None and stationary_kpss is None:
            if stationary_adf:
                score += 0.6  # ADF gets KPSS's 40% + original 20%
            if rolling_stable:
                score += 0.4  # Rolling gets remaining 40%

        # Case 3: ADF N/A (redistribute to KPSS + Rolling)
        elif stationary_adf is None and stationary_kpss is not None:
            if stationary_kpss:
                score += 0.6  # KPSS gets ADF's 40% + original 20%
            if rolling_stable:
                score += 0.4  # Rolling gets remaining 40%

        # Case 4: Both statistical tests N/A (rely on Rolling only)
        else:
            if rolling_stable:
                score = 1.0  # Rolling gets 100% weight
            else:
                score = 0.0

        # Decision: score ≥ 0.5 → stationary
        is_stationary = score >= 0.5

        return is_stationary, float(score)