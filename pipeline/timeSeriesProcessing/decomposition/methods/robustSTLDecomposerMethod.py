"""
Robust time series decomposition method based on STL.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from pipeline.helpers.evaluation.qualityEvaluator import QualityEvaluator
from pipeline.helpers.utils import validate_required_locals
from pipeline.timeSeriesProcessing.decomposition.methods.baseDecomposerMethod import (
    BaseDecomposerMethod,
)

__version__ = "1.1.0"


class RobustSTLDecomposerMethod(BaseDecomposerMethod):
    """
    Robust STL decomposer with improved outlier resistance.

    Extended version of STL with outlier preprocessing,
    iterative refinement and component postprocessing.
    """

    # Standard default configurations for Robust STL
    DEFAULT_CONFIG = {
        **BaseDecomposerMethod.DEFAULT_CONFIG,
        # Main Robust STL characteristics (constants)
        "robust": True,  # Robust LOESS smoothing (Does not require adaptation)
        "robust_weights": True,  # Robust weights (Does not require adaptation)
        "preprocessing_enabled": True,  # Outlier preprocessing (Does not require adaptation)
        "inner_iter": 2,  # LOESS inner iterations (Does not require adaptation)
        "robust_mode": "enhanced",  # Robustness mode (Does not require adaptation)
        # STL window parameters (adapted in configDecomposition)
        # "period": None,  # [AUTO] from main_period
        # "seasonal": 7,  # [AUTO] adaptive seasonal window based on main_period
        # "trend": None,  # [AUTO] based on trend_strength and seasonal
        # Robustness parameters (adapted in configDecomposition)
        # "outlier_threshold": 3.0,  # [AUTO] based on noise_level, outlier_ratio, volatility
        # "max_iterations": 5,  # [AUTO] adaptive based on data quality
        # "convergence_threshold": 0.01,  # [AUTO] adaptive based on volatility
        # LOESS polynomial degrees (adapted in configDecomposition)
        # "seasonal_deg": 1,  # [AUTO] deg=0 for high noise
        # "trend_deg": 1,  # [AUTO] deg=0 for stationary series
        # Robustness iterations (adapted in configDecomposition)
        # "outer_iter": 1,  # [AUTO] adaptive based on noise and data size
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize robust STL method.

        Args:
            config: Configuration with parameters (fully adapted)
        """
        # Merge configuration with Robust STL defaults
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged_config)

        # Validate Robust STL specific parameters
        validate_required_locals(
            [
                "robust",
                "seasonal",
                "trend",
                "outlier_threshold",
                "max_iterations",
                "convergence_threshold",
                "seasonal_deg",
                "trend_deg",
                "robust_weights",
                "inner_iter",
                "outer_iter",
                "preprocessing_enabled",
                "robust_mode",
            ],
            self.config,
        )

        # Cache for iterative process
        self._outlier_cache = {}
        self._last_iteration_count = 1

    def __str__(self) -> str:
        """Standard string representation for Robust STL logging."""
        return (
            f"RobustSTLDecomposerMethod(seasonal={self.config['seasonal']}, "
            f"trend={self.config['trend']}, outlier_threshold={self.config['outlier_threshold']}, "
            f"max_iterations={self.config['max_iterations']})"
        )

    def _validate_config(self) -> None:
        """Validate Robust STL method configuration."""
        # All required parameters already checked in __init__ via validate_required_locals

        # Additional validation of Robust STL parameter logic
        seasonal = self.config["seasonal"]
        trend = self.config["trend"]
        outlier_threshold = self.config["outlier_threshold"]
        max_iterations = self.config["max_iterations"]
        convergence_threshold = self.config["convergence_threshold"]

        # Validate seasonal parameter
        if seasonal is not None and seasonal < 3:
            raise ValueError(
                f"Robust STL seasonal parameter must be >= 3 or None, got {seasonal}"
            )

        # Validate trend parameter
        if trend is not None and trend < 3:
            raise ValueError(
                f"Robust STL trend parameter must be >= 3 or None, got {trend}"
            )

        # Validate outlier_threshold
        if outlier_threshold <= 0:
            raise ValueError(
                f"Robust STL outlier_threshold must be > 0, got {outlier_threshold}"
            )

        # Validate max_iterations
        if max_iterations < 1:
            raise ValueError(
                f"Robust STL max_iterations must be >= 1, got {max_iterations}"
            )

        # Validate convergence_threshold
        if convergence_threshold <= 0 or convergence_threshold >= 1:
            raise ValueError(
                f"Robust STL convergence_threshold must be in (0, 1), got {convergence_threshold}"
            )

        # Validate degree parameters
        for param in ["seasonal_deg", "trend_deg"]:
            if param in self.config:
                value = self.config[param]
                if value is not None and (value < 0 or value > 2):
                    raise ValueError(
                        f"Robust STL {param} must be 0, 1, or 2, got {value}"
                    )

    def process(
        self, data: pd.Series, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform robust STL decomposition.

        Args:
            data: Time series for decomposition
            context: Processing context

        Returns:
            Standardized decomposition result
        """
        try:
            # 1. CRITICAL fail-fast validation
            critical_validation = self.validate_input_critical(data)
            if critical_validation is not None:
                return critical_validation

            # 2. Standard input data validation with decomposition specifics
            validation = self.validate_input(data)
            if validation["status"] == "error":
                return validation

            # Extract context parameters
            context_params = self.extract_context_parameters(context)

            # Period extraction
            period = self.config["period"]

            if not period or period < 2:
                return self.handle_error(
                    ValueError("Period not defined or too small"),
                    "period extraction",
                )

            # Check sufficient data for robust STL
            if len(data) < 2 * period + 1:
                return self.handle_error(
                    ValueError(
                        f"Insufficient data for robust STL with period {period}, "
                        f"got {len(data)} points"
                    ),
                    "data validation",
                )

            logging.info(
                f"{self} - Starting robust STL: length={len(data)}, period={period}"
            )

            # Data preprocessing with outlier and context consideration
            cleaned_data = self._robust_preprocess(data, context_params)

            # Iterative robust decomposition
            trend, seasonal, residual = self._iterative_robust_stl(
                cleaned_data, data, period, context_params
            )

            # Component postprocessing
            trend, seasonal, residual = self._postprocess_components(
                data, trend, seasonal, residual, context_params
            )

            # Prepare method additional data
            additional_data = {
                "period_used": period,
                "seasonal_window": self.config["seasonal"],
                "trend_window": self.config["trend"],
                "outlier_threshold": self.config["outlier_threshold"],
                "preprocessing_applied": self.config["preprocessing_enabled"],
                "iterations_performed": self._last_iteration_count,
                "method_version": __version__,
            }

            # Prepare standardized result through BaseDecomposerMethod
            result = self.prepare_decomposition_result(
                trend=trend,
                seasonal=seasonal,
                residual=residual,
                data=data,
                context_params=context_params,
                additional_data=additional_data,
            )

            logging.info(
                f"{self} - Robust STL completed: period={period}, "
                f"iterations={self._last_iteration_count}, "
                f"quality={result['result']['quality_score']:.3f}"
            )

            return result

        except Exception as e:
            return self.handle_error(e, "Robust STL decomposition")

    def _robust_preprocess(
        self, data: pd.Series, context_params: Dict[str, Any]
    ) -> pd.Series:
        """
        Robust data preprocessing with adaptation to data quality.

        Args:
            data: Original time series
            context_params: Parameters from context with data characteristics

        Returns:
            Cleaned time series
        """
        if not self.config["preprocessing_enabled"]:
            return data

        cleaned_data = self.preprocess_data(data)  # Base preprocessing

        # Adaptive outlier handling based on context
        outlier_ratio = self._extract_outlier_ratio(context_params)
        data_quality_score = self._extract_data_quality_score(context_params)

        # Apply robust filtering only if significant outliers exist
        if outlier_ratio > 0.05 or data_quality_score < 0.7:
            logging.info(
                f"{self} - Applying robust filtering: outlier_ratio={outlier_ratio:.3f}, "
                f"data_quality={data_quality_score:.3f}"
            )

            cleaned_data = self._apply_robust_filtering(cleaned_data, context_params)

        return cleaned_data

    def _apply_robust_filtering(
        self, data: pd.Series, context_params: Dict[str, Any]
    ) -> pd.Series:
        """
        Apply robust outlier filtering with adaptive thresholds.
        Vectorized version for improved performance.

        Args:
            data: Time series for filtering
            context_params: Context parameters

        Returns:
            Filtered time series
        """
        cleaned_data = data.copy()

        # Adaptive window based on data characteristics
        volatility = self._extract_volatility(context_params)
        window_size = self._calculate_adaptive_window(len(data), volatility)

        # Robust statistics - vectorized
        rolling_median = cleaned_data.rolling(
            window=window_size, center=True, min_periods=1
        ).median()

        # Vectorized MAD calculation
        abs_deviations = np.abs(cleaned_data - rolling_median)
        rolling_mad = abs_deviations.rolling(
            window=window_size, center=True, min_periods=1
        ).median()

        # Adaptive outlier threshold
        threshold = self._calculate_adaptive_threshold(context_params)

        # Vectorized outlier detection
        threshold_values = threshold * rolling_mad
        outlier_mask = abs_deviations > threshold_values

        # Vectorized outlier replacement
        if outlier_mask.any():
            # Use numpy.where for efficient replacement
            cleaned_values = np.where(outlier_mask, rolling_median, cleaned_data)
            cleaned_data = pd.Series(cleaned_values, index=data.index)

            outlier_count = int(
                outlier_mask.sum()
            )  # Convert to int for logging
            logging.info(
                f"{self} - Replaced {outlier_count} outliers "
                f"(threshold={threshold:.2f})"
            )

        return cleaned_data

    def _iterative_robust_stl(
        self,
        cleaned_data: pd.Series,
        original_data: pd.Series,
        period: int,
        context_params: Dict[str, Any],
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Iterative robust STL decomposition with adaptive convergence criteria.

        Args:
            cleaned_data: Preprocessed data
            original_data: Original data
            period: Seasonality period
            context_params: Context parameters

        Returns:
            Tuple[trend, seasonality, residuals]
        """
        max_iterations = self.config["max_iterations"]
        convergence_threshold = self._calculate_adaptive_convergence_threshold(
            context_params
        )

        # Initial decomposition
        trend, seasonal, residual = self._perform_single_stl(cleaned_data, period)
        self._last_iteration_count = 1
        prev_residual_std = residual.std()

        # Iterative refinement with early stopping
        for iteration in range(2, max_iterations + 1):
            # Detect outliers in residuals with caching
            cache_key = f"{iteration}_{hash(tuple(residual))}"
            if cache_key in self._outlier_cache:
                residual_outliers = self._outlier_cache[cache_key]
            else:
                residual_outliers = self._detect_residual_outliers(
                    residual, context_params
                )
                self._outlier_cache[cache_key] = residual_outliers

            if residual_outliers.sum() == 0:
                logging.debug(
                    f"{self} - No outliers at iteration {iteration}, stopping"
                )
                break

            # Adjust data
            adjusted_data = original_data.copy()
            adjusted_data[residual_outliers] = (
                trend[residual_outliers] + seasonal[residual_outliers]
            )

            # Re-decompose
            trend, seasonal, residual = self._perform_single_stl(adjusted_data, period)

            # Check convergence with adaptive threshold
            current_residual_std = residual.std()
            improvement = (
                abs(prev_residual_std - current_residual_std) / prev_residual_std
                if prev_residual_std > 0
                else 0
            )

            if improvement < convergence_threshold:
                logging.debug(
                    f"{self} - Convergence reached at iteration {iteration} "
                    f"(improvement={improvement:.6f} < {convergence_threshold:.6f})"
                )
                break

            prev_residual_std = current_residual_std
            self._last_iteration_count = iteration

        return trend, seasonal, residual

    def _perform_single_stl(
        self, data: pd.Series, period: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Perform single iteration of STL decomposition with robust parameters.

        Args:
            data: Time series
            period: Seasonality period

        Returns:
            Tuple[trend, seasonality, residuals]
        """
        # Window parameters from configuration with validation
        seasonal_window = self.config["seasonal"]
        if seasonal_window % 2 == 0:
            seasonal_window += 1

        trend_window = self.config["trend"]
        if trend_window and trend_window % 2 == 0:
            trend_window += 1

        # STL decomposition with robust parameters
        stl = STL(
            data,
            period=period,
            seasonal=seasonal_window,
            trend=trend_window,
            robust=self.config["robust"],
            seasonal_deg=self.config["seasonal_deg"],
            trend_deg=self.config["trend_deg"],
        )

        result = stl.fit(
            inner_iter=self.config["inner_iter"], outer_iter=self.config["outer_iter"]
        )

        return result.trend, result.seasonal, result.resid

    def _detect_residual_outliers(
        self, residual: pd.Series, context_params: Dict[str, Any]
    ) -> pd.Series:
        """
        Detect outliers in residuals with adaptive thresholds.

        Args:
            residual: Decomposition residuals
            context_params: Context parameters

        Returns:
            Outlier mask
        """
        # Robust scale estimation through MAD
        median_residual = residual.median()
        mad = np.median(np.abs(residual - median_residual))

        # Adaptive threshold based on context
        threshold = self._calculate_adaptive_threshold(context_params)

        # Outliers with normal distribution constant
        outliers = np.abs(residual - median_residual) > threshold * mad * 1.4826

        return pd.Series(outliers, index=residual.index)

    def _postprocess_components(
        self,
        original_data: pd.Series,
        trend: pd.Series,
        seasonal: pd.Series,
        residual: pd.Series,
        context_params: Dict[str, Any],
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Postprocess decomposition components with adaptation to data type.

        Args:
            original_data: Original data
            trend: Trend component
            seasonal: Seasonal component
            residual: Residual component
            context_params: Context parameters

        Returns:
            Tuple[processed trend, processed seasonality, processed residuals]
        """
        # Check reconstruction quality
        reconstructed = trend + seasonal
        quality_evaluator = QualityEvaluator(evaluation_type="decomposition")
        mse = quality_evaluator.calculate_mse(original_data, reconstructed)
        original_var = np.var(original_data)

        if mse > 0.1 * original_var:
            logging.warning(
                f"{self} - High reconstruction error: {mse:.6f} "
                f"(10% of variance: {0.1 * original_var:.6f})"
            )

            # Adjustment to improve reconstruction
            adjustment = (original_data - reconstructed).mean()
            trend = trend + adjustment
            residual = original_data - trend - seasonal

        # Adaptation for high volatility data
        volatility = self._extract_volatility(context_params)
        if volatility > 0.7:
            trend, seasonal, residual = self._adjust_for_high_volatility(
                original_data, trend, seasonal, residual, volatility
            )

        return trend, seasonal, residual

    def _adjust_for_high_volatility(
        self,
        original_data: pd.Series,
        trend: pd.Series,
        seasonal: pd.Series,
        residual: pd.Series,
        volatility: float,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Adjust components for high volatility data.

        Args:
            original_data: Original data
            trend: Trend component
            seasonal: Seasonal component
            residual: Residual component
            volatility: Volatility level

        Returns:
            Tuple[adjusted components]
        """
        # Check if seasonality is too large
        seasonal_var = np.var(seasonal)
        original_var = np.var(original_data)

        if seasonal_var > 0.6 * original_var:
            # Damping factor adapts to volatility level
            damping_factor = np.sqrt(0.6 * original_var / seasonal_var)
            damping_factor *= 1 - (volatility - 0.7) * 0.5  # Additional reduction

            seasonal = seasonal * damping_factor
            residual = original_data - trend - seasonal

            logging.info(
                f"{self} - Applied damping factor {damping_factor:.3f} "
                f"to seasonality for volatility={volatility:.3f}"
            )

        return trend, seasonal, residual

    # === Utility methods for extracting context parameters ===

    def _extract_outlier_ratio(self, context_params: Dict[str, Any]) -> float:
        """Extract outlier ratio from context without .get()."""
        analyzer_props = context_params['additional_params']["currentProperties"]["analyzer"]
        return analyzer_props["outlier_ratio"]

    def _extract_data_quality_score(self, context_params: Dict[str, Any]) -> float:
        """Extract data quality score from context without .get()."""
        analyzer_props = context_params['additional_params']["currentProperties"]["analyzer"]
        return analyzer_props["data_quality_score"]

    def _extract_volatility(self, context_params: Dict[str, Any]) -> float:
        """Extract volatility from context without .get()."""
        analyzer_props = context_params['additional_params']["currentProperties"]["analyzer"]
        return analyzer_props["volatility"]

    def _calculate_adaptive_window(self, data_length: int, volatility: float) -> int:
        """Calculate adaptive window size."""
        base_window = min(5, data_length // 10)
        # Increase window for high volatility
        volatility_factor = 1 + volatility
        adaptive_window = int(base_window * volatility_factor)

        # Ensure oddness
        if adaptive_window % 2 == 0:
            adaptive_window += 1

        return max(3, adaptive_window)

    def _calculate_adaptive_threshold(self, context_params: Dict[str, Any]) -> float:
        """Calculate adaptive outlier threshold."""
        base_threshold = self.config["outlier_threshold"]

        # Adapt to data quality
        data_quality = self._extract_data_quality_score(context_params)
        volatility = self._extract_volatility(context_params)

        # Lower threshold for low quality, raise for high volatility
        quality_factor = 2 - data_quality  # 1.0-2.0
        volatility_factor = 1 + volatility * 0.5  # 1.0-1.5

        adaptive_threshold = base_threshold * quality_factor * volatility_factor
        return max(1.5, min(5.0, adaptive_threshold))

    def _calculate_adaptive_convergence_threshold(
        self, context_params: Dict[str, Any]
    ) -> float:
        """Calculate adaptive convergence threshold."""
        base_threshold = self.config["convergence_threshold"]

        # Adapt to data volatility
        volatility = self._extract_volatility(context_params)

        # For high volatility require greater convergence
        volatility_factor = 1 + volatility
        adaptive_threshold = base_threshold * volatility_factor

        return max(0.001, min(0.1, adaptive_threshold))