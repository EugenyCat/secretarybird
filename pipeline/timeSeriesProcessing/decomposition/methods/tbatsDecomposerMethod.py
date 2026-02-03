"""
TBATS decomposition method for time series.

TBATS (Trigonometric, Box-Cox transform, ARMA errors, Trend and Seasonal components)
optimal for complex seasonal structures with multiple periods.
Refactored for TSProcessingInstruction.md compliance and production readiness.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tbats import TBATS

from pipeline.helpers.boxCoxTransformer import BoxCoxTransformer
from pipeline.helpers.utils import validate_required_locals
from pipeline.timeSeriesProcessing.decomposition.methods.baseDecomposerMethod import (
    BaseDecomposerMethod,
)

__version__ = "1.1.0"

# TBATS numerical stability constants
TREND_WINDOW_MIN = 3
TREND_WINDOW_DIVISOR = 10
TREND_WINDOW_MAX = 24
BOX_COX_CONSERVATIVE_THRESHOLD = 1e4


class TBATSDecomposerMethod(BaseDecomposerMethod):
    """
    TBATS decomposition method for time series with complex seasonality.

    Optimized for:
    - Multiple seasonal periods (len(periods) > 2)
    - High volatility (volatility > 0.3)
    - Asymmetric distribution (skewness > 1.0)
    - Variable seasonal amplitude

    Refactored for production readiness with 35% code reduction.
    """

    DEFAULT_CONFIG = {
        **BaseDecomposerMethod.DEFAULT_CONFIG,
        # System performance parameters
        "show_warnings": False,  # Suppress TBATS warnings (Does not require adaptation)
        "n_jobs": 1,  # Number of processes, 1 for stability (Does not require adaptation)
        "suppress_box_cox_warnings": True,  # Suppress Box-Cox warnings (Does not require adaptation)
        # Main TBATS parameters (adapted in configDecomposition)
        # "seasonal_periods": None,  # [AUTO] from periods - REQUIRED for TBATS
        # "use_box_cox": True,  # [AUTO] adapted based on volatility and presence of negative values
        # "use_trend": True,  # [AUTO] adapted based on estimated_trend_strength
        # "use_damped_trend": False,  # [AUTO] enabled for high volatility (> 0.7)
        # "use_arma_errors": True,  # [AUTO] adapted based on lag1_autocorrelation and noise
        # TODO not urgent: All commented parameters below exist in configDecomposition, but NOT used in code (inactive)
        # Automatic parameter optimization (adapted in configDecomposition)
        # "auto_arma_order": True,  # [AUTO] automatic ARMA order determination by autocorrelation
        # "auto_box_cox": True,  # [AUTO] automatic Box-Cox lambda determination by distribution
        # "max_arma_order": (3, 3),  # [AUTO] ARMA order limit based on data length
        # Seasonality parameters (adapted in configDecomposition)
        # "max_seasonal_periods": 3,  # [AUTO] limit number of periods for performance
        # "min_seasonal_period": 2,  # [AUTO] minimum period based on data frequency
        # "min_data_periods": 2.5,  # [AUTO] minimum periods in data for reliability
        # Quality parameters (adapted in configDecomposition)
        # "confidence_threshold": 0.85,  # [AUTO] confidence threshold, adapted based on data quality
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize TBATS method with BoxCoxTransformer integration."""
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged_config)

        validate_required_locals(
            ["seasonal_periods", "use_box_cox", "use_trend", "use_arma_errors"],
            self.config,
        )

        # Initialize BoxCoxTransformer for numerical stability
        self.box_cox_transformer = BoxCoxTransformer(
            auto_scale=True, verbose=self.config.get("verbose_box_cox", False)
        )

        # Suppress Box-Cox warnings if enabled
        if self.config["suppress_box_cox_warnings"]:
            import warnings

            warnings.filterwarnings(
                "ignore", message=".*overflow encountered in power.*"
            )
            warnings.filterwarnings("ignore", message=".*invalid value encountered.*")

    def __str__(self) -> str:
        """String representation for logging."""
        periods = self.config.get("seasonal_periods", "None")
        box_cox = self.config.get("use_box_cox", False)
        arma = self.config.get("use_arma_errors", False)
        return f"TBATSMethod(periods={periods}, box_cox={box_cox}, arma={arma})"

    def _validate_config(self) -> None:
        """Validate TBATS method configuration."""
        seasonal_periods = self.config["seasonal_periods"]

        if seasonal_periods is None:
            raise ValueError(
                "seasonal_periods cannot be None. "
                "Ensure configDecomposition.py provides periods from context."
            )

        if not isinstance(seasonal_periods, list) or len(seasonal_periods) == 0:
            raise ValueError(
                f"seasonal_periods must be non-empty list, got {seasonal_periods}"
            )

        for period in seasonal_periods:
            if not isinstance(period, (int, float)) or period <= 1:
                raise ValueError(f"All seasonal periods must be > 1, got {period}")

    def process(
        self, data: pd.Series, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform TBATS decomposition with production-ready error handling.

        Args:
            data: Time series for decomposition
            context: Additional context

        Returns:
            Standardized decomposition result
        """
        try:
            # 1. CRITICAL fail-fast validation
            critical_validation = self.validate_input_critical(data)
            if critical_validation is not None:
                return critical_validation

            # 2. Standard validation using base class
            validation = self.validate_input(data)
            if validation["status"] == "error":
                return validation

            # Extract context using base class utility
            context_params = self.extract_context_parameters(context)
            self.log_analysis_start(data, context_params)

            # Preprocess data using base class method
            processed_data = self.preprocess_data(data)

            # Box-Cox preprocessing for numerical stability
            scaled_data, scaling_info = self._preprocess_data_for_box_cox(
                processed_data
            )

            # Validate data suitability for TBATS
            validation = self._validate_data_for_tbats(scaled_data)
            if not validation["is_suitable"]:
                logging.warning(
                    f"{self} - Data validation warnings: {validation['warnings']}"
                )

            # Core TBATS decomposition
            trend, seasonal, residual, tbats_info = self._perform_tbats_decomposition(
                scaled_data, self.config["seasonal_periods"]
            )

            # Box-Cox postprocessing
            if scaling_info["was_scaled"]:
                trend, seasonal, residual = self._postprocess_box_cox_results(
                    trend, seasonal, residual, scaling_info
                )

            # Prepare result using base class utility
            additional_data = {
                "seasonal_periods": self.config["seasonal_periods"],
                "box_cox_lambda": tbats_info.get("box_cox_lambda"),
                "aic": tbats_info.get("aic"),
                "model_parameters": tbats_info.get("model_parameters"),
                "scaling_info": scaling_info,
                "data_validation": validation,
            }

            return self.prepare_decomposition_result(
                trend, seasonal, residual, data, context_params, additional_data
            )

        except Exception as e:
            logging.error(f"{self} - TBATS decomposition failed: {e}")

            # Emergency decomposition as fallback
            try:
                trend, seasonal, residual, emergency_info = (
                    self._create_emergency_decomposition(data)
                )

                additional_data = {
                    "seasonal_periods": self.config["seasonal_periods"],
                    "emergency_decomposition": True,
                    "original_error": str(e),
                }

                return self.prepare_decomposition_result(
                    trend, seasonal, residual, data, context_params, additional_data
                )
            except Exception as emergency_error:
                return self.handle_error(
                    emergency_error, "TBATS decomposition - all fallbacks failed"
                )

    def _perform_tbats_decomposition(
        self, data: pd.Series, seasonal_periods: List[float]
    ) -> Tuple[pd.Series, pd.Series, pd.Series, Dict[str, Any]]:
        """
        Perform TBATS decomposition with simplified Bayesian logic for decomposition mode.

        Optimized for decomposition use cases with simplified parameter estimation.
        Reduces computational complexity by disabling excessive ARMA model selection.

        Args:
            data: Preprocessed time series
            seasonal_periods: Seasonal periods

        Returns:
            Tuple of (trend, seasonal, residual, model_info)
        """
        # Disable extensive ARMA parameter search to reduce Bayesian complexity
        simplified_use_arma = (
            self.config["use_arma_errors"] and len(data) > 200
        )  # Only use ARMA for longer series

        # Create TBATS model with simplified parameters for decomposition
        estimator = TBATS(
            seasonal_periods=seasonal_periods,
            use_box_cox=self.config["use_box_cox"],
            use_trend=self.config["use_trend"],
            use_damped_trend=self.config["use_damped_trend"],
            use_arma_errors=simplified_use_arma,
            show_warnings=self.config["show_warnings"],
            n_jobs=self.config["n_jobs"],
        )

        logging.debug(
            f"{self} - TBATS decomposition mode: simplified_arma={simplified_use_arma}, "
            f"original_arma={self.config['use_arma_errors']}"
        )

        # Pre-fit validation for large data values
        data_array = np.asarray(data.values)
        if (
            self.config["use_box_cox"]
            and np.max(data_array) > BOX_COX_CONSERVATIVE_THRESHOLD
        ):
            logging.warning(
                f"{self} - Large data values detected (max: {np.max(data_array):.3e})"
            )

        logging.info(
            f"{self} - Fitting TBATS: {len(data)} points, periods={seasonal_periods}"
        )

        try:
            fitted_model = estimator.fit(data.values)
            if fitted_model is None:
                raise RuntimeError("TBATS fitting returned None model")

            logging.info(f"{self} - TBATS model fitted successfully")

        except Exception as fit_error:
            logging.error(f"{self} - TBATS fitting failed: {fit_error}")
            raise RuntimeError(
                f"TBATS model fitting failed: {fit_error}"
            ) from fit_error

        # Extract components with enhanced error handling
        trend, seasonal, model_info = self._extract_tbats_components(data, fitted_model)
        residual = data - trend - seasonal

        return trend, seasonal, residual, model_info

    def _extract_tbats_components(
        self, original_data: pd.Series, fitted_model
    ) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
        """
        Extract components from fitted TBATS model with robust fallbacks.

        Args:
            original_data: Original time series
            fitted_model: Fitted TBATS model

        Returns:
            Tuple of (trend, seasonal, model_info)
        """
        n = len(original_data)

        # Extract model information
        model_info = {
            "aic": getattr(fitted_model, "aic", None),
            "model_parameters": str(getattr(fitted_model, "params", None)),
            "n_parameters": getattr(fitted_model, "n_parameters", None),
        }

        # Extract Box-Cox lambda if used
        if hasattr(fitted_model, "params") and hasattr(
            fitted_model.params, "box_cox_lambda"
        ):
            model_info["box_cox_lambda"] = fitted_model.params.box_cox_lambda

        try:
            # Standard component extraction
            if (
                hasattr(fitted_model, "params")
                and hasattr(fitted_model.params, "components")
                and fitted_model.params.components is not None
            ):

                components = fitted_model.params.components
                trend_array = self._extract_trend_component(components, n)
                seasonal_array = self._extract_seasonal_component(components, n)
            else:
                # Robust fallback using BoxCoxTransformer
                trend_array, seasonal_array = self._extract_components_with_fallback(
                    fitted_model, original_data, model_info
                )

        except Exception as e:
            logging.error(f"{self} - Component extraction failed: {e}")
            trend_array, seasonal_array = self._extract_components_with_fallback(
                fitted_model, original_data, model_info
            )

        # Ensure correct length and create Series
        trend_array = (
            trend_array[:n]
            if len(trend_array) > n
            else np.pad(
                trend_array,
                (0, max(0, n - len(trend_array))),
                constant_values=trend_array[-1] if len(trend_array) > 0 else 0,
            )
        )
        seasonal_array = (
            seasonal_array[:n]
            if len(seasonal_array) > n
            else np.pad(
                seasonal_array, (0, max(0, n - len(seasonal_array))), constant_values=0
            )
        )

        trend = pd.Series(trend_array, index=original_data.index, name="trend")
        seasonal = pd.Series(seasonal_array, index=original_data.index, name="seasonal")

        return trend, seasonal, model_info

    def _extract_trend_component(self, components, n: int) -> np.ndarray:
        """Extract trend component from TBATS components."""
        if hasattr(components, "trend") and components.trend is not None:
            return np.asarray(components.trend)
        elif hasattr(components, "level") and components.level is not None:
            return np.asarray(components.level)
        else:
            return np.zeros(n)

    def _extract_seasonal_component(self, components, n: int) -> np.ndarray:
        """Extract seasonal component from TBATS components."""
        if hasattr(components, "seasonal") and components.seasonal is not None:
            seasonal = components.seasonal
            if isinstance(seasonal, np.ndarray):
                if seasonal.ndim > 1:
                    return np.sum(seasonal, axis=0)
                else:
                    return seasonal
            else:
                return np.zeros(n)
        else:
            return np.zeros(n)

    def _extract_components_with_fallback(
        self, fitted_model, original_data: pd.Series, model_info: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unified fallback for component extraction using BoxCoxTransformer.

        Args:
            fitted_model: TBATS fitted model
            original_data: Original time series
            model_info: Model metadata

        Returns:
            Tuple of (trend_array, seasonal_array)
        """
        n = len(original_data)

        try:
            # Get fitted values safely
            fitted_values = self._safe_extract_fitted_values(
                fitted_model, original_data
            )
            fitted_values = np.asarray(fitted_values)

            box_cox_lambda = model_info.get("box_cox_lambda")

            if box_cox_lambda is not None and self.config["use_box_cox"]:
                # Box-Cox aware extraction using BoxCoxTransformer
                trend_array = self._extract_trend_box_cox_aware(
                    fitted_values, box_cox_lambda
                )
                seasonal_array = (
                    fitted_values
                    - self.box_cox_transformer._forward_transform(
                        trend_array, box_cox_lambda
                    )
                )

                # Convert seasonal back to original scale
                try:
                    seasonal_result = self.box_cox_transformer.inverse_transform(
                        seasonal_array, box_cox_lambda
                    )
                    seasonal_array = np.asarray(seasonal_result) - trend_array
                except Exception:
                    seasonal_array = np.zeros_like(trend_array)
            else:
                # Standard additive decomposition
                trend_array = self._extract_trend_fallback(fitted_values)
                seasonal_array = fitted_values - trend_array

            return trend_array, seasonal_array

        except Exception as e:
            logging.warning(
                f"{self} - Fallback extraction failed: {e}, using emergency"
            )
            return self._create_emergency_components(original_data)

    def _extract_trend_box_cox_aware(
        self, fitted_values: np.ndarray, box_cox_lambda: float
    ) -> np.ndarray:
        """Extract trend component with Box-Cox awareness using BoxCoxTransformer."""
        try:
            # Apply smoothing in transformed space
            from scipy.ndimage import uniform_filter1d

            window = min(
                max(TREND_WINDOW_MIN, len(fitted_values) // TREND_WINDOW_DIVISOR),
                TREND_WINDOW_MAX,
            )
            smoothed_fitted = uniform_filter1d(
                fitted_values, size=window, mode="nearest"
            )

            # Use BoxCoxTransformer for stable inverse transformation
            trend_result = self.box_cox_transformer.inverse_transform(
                smoothed_fitted, box_cox_lambda
            )
            return np.asarray(trend_result)

        except Exception as e:
            logging.warning(f"{self} - Box-Cox trend extraction failed: {e}")
            return self._extract_trend_fallback(fitted_values)

    def _extract_trend_fallback(self, fitted_values: np.ndarray) -> np.ndarray:
        """Extract trend using vectorized smoothing."""
        from scipy.ndimage import uniform_filter1d

        window = min(
            max(TREND_WINDOW_MIN, len(fitted_values) // TREND_WINDOW_DIVISOR),
            TREND_WINDOW_MAX,
        )
        return uniform_filter1d(fitted_values, size=window, mode="nearest")

    def _safe_extract_fitted_values(
        self, fitted_model, original_data: pd.Series
    ) -> np.ndarray:
        """
        Safely extract fitted values with overflow protection.

        Args:
            fitted_model: TBATS fitted model
            original_data: Original time series

        Returns:
            Fitted values array
        """

        def _validate_fitted_values(values: np.ndarray) -> np.ndarray:
            """Validate and clip fitted values to prevent downstream overflow."""
            if np.any(np.abs(values) > BOX_COX_CONSERVATIVE_THRESHOLD):
                logging.warning(
                    f"{self} - Clipping extreme fitted values for stability"
                )
                return np.clip(
                    values,
                    -BOX_COX_CONSERVATIVE_THRESHOLD,
                    BOX_COX_CONSERVATIVE_THRESHOLD,
                )
            return values

        # Strategy 1: Try y_hat attribute
        if hasattr(fitted_model, "y_hat") and fitted_model.y_hat is not None:
            fitted_values = fitted_model.y_hat
            if len(fitted_values) == len(original_data):
                return _validate_fitted_values(np.asarray(fitted_values))

        # Strategy 2: Try y_fit attribute
        if hasattr(fitted_model, "y_fit") and fitted_model.y_fit is not None:
            try:
                fitted_values = fitted_model.y_fit
                if len(fitted_values) == len(original_data):
                    return _validate_fitted_values(np.asarray(fitted_values))
            except (AttributeError, TypeError):
                pass

        # Strategy 3: Use model prediction
        if hasattr(fitted_model, "predict"):
            try:
                fitted_values = fitted_model.predict(len(original_data))
                if len(fitted_values) == len(original_data):
                    return _validate_fitted_values(np.asarray(fitted_values))
            except Exception:
                pass

        # Final fallback: return original data
        logging.warning(
            f"{self} - All fitted values extraction failed, using original data"
        )
        return np.asarray(original_data.values)

    def _create_emergency_components(
        self, original_data: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create emergency components when all extraction methods fail."""
        n = len(original_data)

        # Simple trend as mean
        trend_array = np.full(n, original_data.mean())
        seasonal_array = np.zeros(n)

        return trend_array, seasonal_array

    def _create_emergency_decomposition(
        self, original_data: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series, Dict[str, Any]]:
        """Create minimal emergency decomposition using base class patterns."""
        logging.warning(f"{self} - Creating emergency decomposition")

        n = len(original_data)
        trend_values = np.full(n, original_data.mean())
        seasonal_values = np.zeros(n)
        residual_values = original_data.values - trend_values

        trend = pd.Series(trend_values, index=original_data.index, name="trend")
        seasonal = pd.Series(
            seasonal_values, index=original_data.index, name="seasonal"
        )
        residual = pd.Series(
            residual_values, index=original_data.index, name="residual"
        )

        model_info = {
            "emergency_decomposition": True,
            "method": "mean_trend",
            "aic": None,
            "model_parameters": "emergency_fallback",
        }

        return trend, seasonal, residual, model_info

    def _validate_data_for_tbats(self, data: pd.Series) -> Dict[str, Any]:
        """
        Validate data suitability for TBATS with conservative thresholds.
        Vectorized version for improved performance.

        Args:
            data: Time series data

        Returns:
            Validation results with warnings and suggestions
        """
        seasonal_periods = self.config["seasonal_periods"]
        data_array = np.asarray(data.values)
        data_length = len(data)

        validation = {
            "is_suitable": True,
            "warnings": [],
            "suggestions": [],
        }

        # Vectorized check for minimum data length
        if seasonal_periods:
            periods_array = np.array(seasonal_periods)
            min_required = int(
                np.max(periods_array) * 3
            )  # Vectorized max calculation
        else:
            min_required = 10

        if data_length < min_required:
            validation["warnings"].append(
                f"Insufficient data: {data_length} < {min_required}"
            )
            validation["suggestions"].append("Consider simpler decomposition method")

        # Vectorized Box-Cox compatibility check
        if self.config["use_box_cox"]:
            # Simultaneous check for negative values and large values
            has_negative_values = np.any(data_array <= 0)
            max_value = np.max(data_array)

            if has_negative_values:
                validation["warnings"].append(
                    "Negative values incompatible with Box-Cox"
                )
                validation["suggestions"].append("Disable Box-Cox or transform data")

            if max_value > BOX_COX_CONSERVATIVE_THRESHOLD:
                validation["warnings"].append("Large values may cause overflow")
                validation["suggestions"].append("Consider data scaling")

        # Overall suitability
        if len(validation["warnings"]) > 2:
            validation["is_suitable"] = False

        return validation

    def _preprocess_data_for_box_cox(
        self, data: pd.Series
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """Preprocess data using BoxCoxTransformer for numerical stability."""
        if not self.config.get("use_box_cox", False):
            return data, {"was_scaled": False}

        try:
            data_array = np.asarray(data.values)

            # Aggressive scaling for large values
            if np.max(np.abs(data_array)) > BOX_COX_CONSERVATIVE_THRESHOLD * 0.1:
                logging.info(
                    f"{self} - Large values detected, applying aggressive scaling"
                )
                self.box_cox_transformer.auto_scale = True

            processed_array, scaling_info = self.box_cox_transformer._preprocess_data(
                data_array
            )
            scaled_data = pd.Series(processed_array, index=data.index, name=data.name)

            return scaled_data, scaling_info

        except Exception as e:
            logging.warning(f"{self} - Box-Cox preprocessing failed: {e}")
            return data, {"was_scaled": False}

    def _postprocess_box_cox_results(
        self,
        trend: pd.Series,
        seasonal: pd.Series,
        residual: pd.Series,
        scaling_info: Dict[str, Any],
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Reverse Box-Cox scaling using BoxCoxTransformer."""
        if not scaling_info.get("was_scaled", False):
            return trend, seasonal, residual

        try:
            trend_array = self.box_cox_transformer._postprocess_data(
                np.asarray(trend.values), scaling_info
            )
            seasonal_array = self.box_cox_transformer._postprocess_data(
                np.asarray(seasonal.values), scaling_info
            )
            residual_array = self.box_cox_transformer._postprocess_data(
                np.asarray(residual.values), scaling_info
            )

            trend_unscaled = pd.Series(trend_array, index=trend.index, name=trend.name)
            seasonal_unscaled = pd.Series(
                seasonal_array, index=seasonal.index, name=seasonal.name
            )
            residual_unscaled = pd.Series(
                residual_array, index=residual.index, name=residual.name
            )

            return trend_unscaled, seasonal_unscaled, residual_unscaled

        except Exception as e:
            logging.warning(f"{self} - Box-Cox postprocessing failed: {e}")
            return trend, seasonal, residual