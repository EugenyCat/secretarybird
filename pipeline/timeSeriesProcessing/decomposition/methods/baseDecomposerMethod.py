"""
Base class for time series decomposition methods.

Inherits common functionality from BaseTimeSeriesMethod.
Defines specific interface and functionality for decomposition methods.
"""

import logging
from abc import abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from pipeline.helpers.evaluation.qualityEvaluator import QualityEvaluator
from pipeline.timeSeriesProcessing.baseModule.baseMethod import BaseTimeSeriesMethod
from pipeline.helpers.utils import validate_required_locals

__version__ = "1.1.0"


# Data validation constants
MIN_DATA_LENGTH = 10  # Minimum data points for meaningful decomposition
INTERPOLATION_LIMIT_DIVISOR = 4  # Max interpolation as fraction of data length
INTERPOLATION_LIMIT_MAX = 10  # Absolute maximum interpolation points

# Data preprocessing constants
ZERO_REPLACEMENT_RATIO = 0.01  # Ratio for replacing zeros in multiplicative mode
ZERO_REPLACEMENT_FALLBACK = 1e-6  # Fallback when no positive values exist

# Quality assessment constants
PERFECT_RECONSTRUCTION_ERROR = 0.0  # Error for perfect constant data reconstruction
FAILED_RECONSTRUCTION_PENALTY = 10.0  # High penalty for failed constant reconstruction

# Component strength defaults
DEFAULT_TREND_STRENGTH = 0.0
DEFAULT_SEASONAL_STRENGTH = 0.0
DEFAULT_RESIDUAL_STRENGTH = 1.0

# Quality score calculation constants
NOISE_PENALTY_WEIGHT = 0.5  # Weight for noise penalty in quality score
MAX_RECONSTRUCTION_PENALTY = 1.0  # Maximum reconstruction error penalty
MIN_QUALITY_SCORE = 0.0  # Minimum quality score
MAX_QUALITY_SCORE = 1.0  # Maximum quality score
FALLBACK_QUALITY_SCORE = 0.0  # Fallback when quality calculation fails

# Parameter adjustment constants
DATA_LENGTH_NORMALIZATION_FACTOR = 100.0  # Factor for data length normalization
MIN_WINDOW_SIZE = 3  # Minimum window size for any method
ADJUSTMENT_BASE_MULTIPLIER = 1.0  # Base multiplier for parameter adjustment

# Volatility and smoothing bounds
MIN_VOLATILITY_ADJUSTMENT = 0.1  # Minimum volatility for adjustment
MAX_VOLATILITY_ADJUSTMENT = 2.0  # Maximum volatility for adjustment
MIN_SMOOTHING_LEVEL = 0.01  # Minimum smoothing level
MAX_SMOOTHING_LEVEL = 0.99  # Maximum smoothing level

# Seasonal period validation
PERIOD_LENGTH_DIVISOR = 3  # Max period as fraction of data length
DEFAULT_FALLBACK_PERIOD = 7  # Default period when validation fails

# Vectorized adjustment weights for characteristics
# [seasonality_strength, trend_strength, volatility, length_factor]
CHARACTERISTIC_WEIGHTS = [0.3, 0.2, -0.1, 0.4]

# Validation optimization constants (fail-fast)
CRITICAL_ERROR_IMMEDIATE_FAIL = True  # Fail immediately on critical errors
MAX_MISSING_DATA_RATIO = 0.8  # Maximum ratio of missing data before failing
MIN_UNIQUE_VALUES = 3  # Minimum unique values for meaningful decomposition
MAX_ZERO_VARIANCE_RATIO = 0.95  # Maximum ratio of zero variance before failing

# Method-specific tolerances for trend + seasonal + residual = original validation
CONSISTENCY_TOLERANCES = {
    "robust_stl": 0.02,  # High precision for robust baseline method
    "mstl": 0.02,  # Slightly higher due to iterative nature
    "ssa": 0.025,  # SVD approximation inherent errors
    "nbeats": 0.02,  # Neural network approximation errors
    "fourier": 0.015,  # FFT reconstruction precision
    "tbats": 0.03,  # Box-Cox + ARMA modeling errors
    "prophet": 0.025,  # Bayesian posterior uncertainty
}
DEFAULT_CONSISTENCY_TOLERANCE = 0.02  # Fallback tolerance for unknown methods
HIGH_RECONSTRUCTION_ERROR_THRESHOLD = 0.1  # Threshold for warning messages
CONSISTENCY_CHECK_ENABLED = True  # Enable/disable consistency validation


class BaseDecomposerMethod(BaseTimeSeriesMethod):
    """
    Base class for time series decomposition methods.

    Inherits common functionality from BaseTimeSeriesMethod.
    Adds specific logic for time series decomposition.
    """

    # Standard default configurations for decomposition
    DEFAULT_CONFIG = {
        **BaseTimeSeriesMethod.DEFAULT_CONFIG,
        "min_data_length": MIN_DATA_LENGTH,  # Minimum for correct decomposition (Does not require adaptation)
        # "model": "additive",  # Default decomposition type (adapted in configDecomposition)
        # "max_missing_ratio": 0.5,  # Tolerance for missing values (adapted in configDecomposition)
        # "confidence_threshold": 0.1,  # Component quality threshold (adapted in configDecomposition)
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize decomposition method.

        Args:
            config: Method configuration (must be fully adapted)

        Raises:
            ValueError: If configuration is missing or invalid
        """
        if not config:
            raise ValueError(
                f"Configuration is required for {self.__class__.__name__}. "
                f"Use build_config_from_properties() to generate configuration."
            )

        # Merge configuration with decomposition defaults
        merged_config = {**self.DEFAULT_CONFIG, **config}
        super().__init__(merged_config)

        # Validate base decomposition parameters
        validate_required_locals(["confidence_threshold"], self.config)

        # Extract base parameters
        self.confidence_threshold = self.config["confidence_threshold"]
        self.model_type = self.config["model"]

        # Validate method configuration (must be implemented in subclasses)
        self._validate_config()

    def __str__(self) -> str:
        """Standard string representation for decomposition logging."""
        return (
            f"{self.name}(model={self.model_type}, "
            f"confidence={self.confidence_threshold})"
        )

    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate method configuration.

        Must be implemented in subclasses to check
        method-specific parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    def process(
        self, data: pd.Series, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform time series decomposition.

        Args:
            data: Time series for decomposition
            context: Context with additional information

        Returns:
            Dict with standard format:
            {
                'status': 'success/error',
                'result': {
                    'trend': pd.Series,
                    'seasonal': pd.Series,
                    'residual': pd.Series,
                    'quality_score': float,
                    'quality_metrics': dict,
                    ...
                },
                'metadata': {...}
            }
        """
        pass

    def validate_input(
        self, data: pd.Series, min_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extended input data validation considering decomposition specifics.

        Args:
            data: Time series for validation
            min_length: Minimum data length

        Returns:
            Dict with validation result
        """
        # Minimum length for decomposition
        min_len = min_length or self.config["min_data_length"]

        # Use base validation
        validation_result = super().validate_input(data, min_len)
        if validation_result["status"] == "error":
            return validation_result

        # Additional checks for decomposition
        try:
            # Check for constancy (critical for decomposition)
            if data.nunique() <= 1:
                return self._create_error_response(
                    "Data is constant, decomposition not possible"
                )

            # Check for sufficient variability
            if data.nunique() < 3:
                return self._create_error_response(
                    f"Insufficient variability: only {data.nunique()} unique values"
                )

            # Check for multiplicative decomposition
            if self.model_type == "multiplicative" and (data <= 0).any():
                return self._create_error_response(
                    "Multiplicative decomposition requires all positive values"
                )

            # Check for extreme values
            if np.isinf(data).any():
                return self._create_error_response("Data contains infinite values")

            return {"status": "success"}

        except Exception as e:
            return self._create_error_response(
                f"Decomposition validation error: {str(e)}"
            )

    def extract_context_parameters(
        self, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract context parameters considering decomposition specifics.

        Args:
            context: Execution context

        Returns:
            Dict with extracted parameters
        """
        # Get base parameters
        base_params = super().extract_context_parameters(context)

        # Add decomposition-specific parameters
        if context:
            base_params["model_type"] = self.model_type

        return base_params

    def preprocess_data(self, data: pd.Series) -> pd.Series:
        """
        Preprocess data before decomposition.

        Base implementation handles missing values and zeros for multiplicative model.

        Args:
            data: Original time series

        Returns:
            Preprocessed time series
        """
        processed = data.copy()

        # Handle missing values
        if processed.isna().any():
            n_missing = processed.isna().sum()
            missing_ratio = n_missing / len(processed)

            logging.info(
                f"{self} - Filling {n_missing} missing values ({missing_ratio:.1%}) "
                f"with interpolation"
            )

            # Linear interpolation with limits
            processed = processed.interpolate(
                method="linear",
                limit_direction="both",
                limit=min(
                    INTERPOLATION_LIMIT_MAX,
                    len(processed) // INTERPOLATION_LIMIT_DIVISOR,
                ),
            )

            # If still missing values, fill with forward/backward fill
            if processed.isna().any():
                processed = processed.ffill().bfill()
                logging.warning(f"{self} - Some gaps filled with forward/backward fill")

        # Handle zeros for multiplicative model
        if self.model_type == "multiplicative":
            zero_mask = processed <= 0
            if zero_mask.any():
                min_positive = processed[processed > 0].min()
                replacement_value = (
                    min_positive * ZERO_REPLACEMENT_RATIO
                    if min_positive > 0
                    else ZERO_REPLACEMENT_FALLBACK
                )
                processed[zero_mask] = replacement_value
                logging.warning(
                    f"{self} - Replaced {zero_mask.sum()} non-positive values "
                    f"with {replacement_value} for multiplicative decomposition"
                )

        return processed

    def validate_input_critical(self, data: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Critical fail-fast validation for decomposition input data.

        Performs comprehensive validation and fails immediately on critical errors
        to avoid wasting computational resources on invalid data.

        Args:
            data: Input time series data

        Returns:
            None if validation passes, error dict if critical failure
        """
        if not CRITICAL_ERROR_IMMEDIATE_FAIL:
            return None  # Skip critical validation if disabled

        try:
            # 1. Critical: Check for completely empty data
            if len(data) == 0:
                return self.handle_error(
                    ValueError("Critical: Empty time series data"),
                    "critical_validation",
                )

            # 2. Critical: Check minimum data length
            if len(data) < MIN_DATA_LENGTH:
                return self.handle_error(
                    ValueError(
                        f"Critical: Data length {len(data)} < minimum required {MIN_DATA_LENGTH}"
                    ),
                    "critical_validation",
                )

            # 3. Critical: Check excessive missing data
            missing_ratio = data.isna().sum() / len(data)
            if missing_ratio > MAX_MISSING_DATA_RATIO:
                return self.handle_error(
                    ValueError(
                        f"Critical: Missing data ratio {missing_ratio:.2%} > maximum {MAX_MISSING_DATA_RATIO:.2%}"
                    ),
                    "critical_validation",
                )

            # 4. Critical: Check for sufficient unique values
            valid_data = data.dropna()
            if len(valid_data) > 0:
                unique_values = valid_data.nunique()
                if unique_values < MIN_UNIQUE_VALUES:
                    return self.handle_error(
                        ValueError(
                            f"Critical: Only {unique_values} unique values, minimum {MIN_UNIQUE_VALUES} required"
                        ),
                        "critical_validation",
                    )

                # 5. Critical: Check for excessive zero variance (constant data)
                if len(valid_data) > 1:
                    variance = valid_data.var()
                    if variance == 0 or pd.isna(variance):
                        return self.handle_error(
                            ValueError(
                                "Critical: Time series has zero variance (constant data)"
                            ),
                            "critical_validation",
                        )

                    # Check for near-constant data (mostly same values)
                    mode_count = (
                        valid_data.mode().iloc[0] if len(valid_data.mode()) > 0 else 0
                    )
                    mode_ratio = (valid_data == mode_count).sum() / len(valid_data)
                    if mode_ratio > MAX_ZERO_VARIANCE_RATIO:
                        return self.handle_error(
                            ValueError(
                                f"Critical: {mode_ratio:.2%} of data has same value (near-constant)"
                            ),
                            "critical_validation",
                        )

            # 6. Critical: Check for infinite or extremely large values
            if not data.isna().all():
                finite_data = data[data.notna()]
                if not np.isfinite(finite_data).all():
                    return self.handle_error(
                        ValueError(
                            "Critical: Time series contains infinite or NaN values"
                        ),
                        "critical_validation",
                    )

                # Check for extremely large values that could cause numerical instability
                max_abs_value = np.abs(finite_data).max()
                if max_abs_value > 1e10:  # Arbitrary large number threshold
                    return self.handle_error(
                        ValueError(
                            f"Critical: Extremely large values detected (max: {max_abs_value:.2e})"
                        ),
                        "critical_validation",
                    )

            logging.debug(
                f"{self} - Critical validation passed for {len(data)} data points"
            )
            return None  # Validation passed

        except Exception as e:
            logging.error(f"{self} - Critical validation failed with exception: {e}")
            return self.handle_error(e, "critical_validation")

    def _validate_component_consistency(
        self,
        trend: pd.Series,
        seasonal: pd.Series,
        residual: pd.Series,
        original: pd.Series,
    ) -> Dict[str, Any]:
        """
        Universal consistency validation for decomposition components.

        Validates that trend + seasonal + residual ≈ original within method-specific tolerance.
        Provides unified warning system for high reconstruction errors across all methods.

        Args:
            trend: Trend component
            seasonal: Seasonal component
            residual: Residual component
            original: Original time series

        Returns:
            Dict with validation results:
            {
                'is_valid': bool,
                'error': float,  # Normalized error
                'tolerance_used': float,
                'method_type': str,
                'warnings': List[str]
            }
        """
        if not CONSISTENCY_CHECK_ENABLED:
            return {
                "is_valid": True,
                "error": 0.0,
                "tolerance_used": 0.0,
                "method_type": self.model_type,
                "warnings": [],
            }

        try:
            # Align all components to same length/index
            min_len = min(len(trend), len(seasonal), len(residual), len(original))
            trend_aligned = trend.iloc[:min_len]
            seasonal_aligned = seasonal.iloc[:min_len]
            residual_aligned = residual.iloc[:min_len]
            original_aligned = original.iloc[:min_len]

            # Calculate reconstruction: trend + seasonal + residual
            reconstruction = trend_aligned + seasonal_aligned + residual_aligned

            # Calculate normalized consistency error
            absolute_error = np.abs(reconstruction - original_aligned)
            normalized_error = np.mean(absolute_error) / (
                np.std(original_aligned) + 1e-10
            )

            # Get method-specific tolerance based on class name
            class_name = self.__class__.__name__.lower()
            # Map class names to tolerance keys
            if "robuststl" in class_name:
                method_key = "robust_stl"
            elif "mstl" in class_name:
                method_key = "mstl"
            elif "ssa" in class_name:
                method_key = "ssa"
            elif "nbeats" in class_name:
                method_key = "nbeats"
            elif "fourier" in class_name:
                method_key = "fourier"
            elif "tbats" in class_name:
                method_key = "tbats"
            elif "prophet" in class_name:
                method_key = "prophet"
            else:
                method_key = "unknown"

            tolerance = CONSISTENCY_TOLERANCES.get(
                method_key, DEFAULT_CONSISTENCY_TOLERANCE
            )

            # Validation result
            is_valid = normalized_error <= tolerance
            warnings = []

            # Unified warning system for high reconstruction error
            if normalized_error > HIGH_RECONSTRUCTION_ERROR_THRESHOLD:
                warning_msg = (
                    f"High reconstruction error detected: {normalized_error:.4f} "
                    f"(tolerance: {tolerance:.4f}) for method {method_key}"
                )
                warnings.append(warning_msg)
                logging.warning(f"{self} - {warning_msg}")

            # Additional warning for consistency failure
            if not is_valid:
                consistency_msg = (
                    f"Component consistency validation FAILED: "
                    f"error={normalized_error:.4f} > tolerance={tolerance:.4f} for {method_key}"
                )
                warnings.append(consistency_msg)
                logging.error(f"{self} - {consistency_msg}")
            else:
                logging.debug(
                    f"{self} - Component consistency validation PASSED ({method_key}): "
                    f"error={normalized_error:.4f} <= tolerance={tolerance:.4f}"
                )

            return {
                "is_valid": is_valid,
                "error": float(normalized_error),
                "tolerance_used": float(tolerance),
                "method_key": method_key,  # Algorithm-specific key (e.g. 'robust_stl', 'ssa')
                "method_type": self.model_type,  # Model type (e.g. 'additive', 'multiplicative')
                "warnings": warnings,
            }

        except Exception as e:
            error_msg = f"Component consistency validation failed with exception: {e}"
            logging.error(f"{self} - {error_msg}")
            return {
                "is_valid": False,
                "error": float("inf"),
                "tolerance_used": 0.0,
                "method_type": self.model_type,
                "warnings": [error_msg],
            }

    def prepare_decomposition_result(
        self,
        trend: pd.Series,
        seasonal: pd.Series,
        residual: pd.Series,
        data: pd.Series,
        context_params: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Prepare standardized decomposition result.

        Args:
            trend: Trend component
            seasonal: Seasonal component
            residual: Residual component
            data: Original time series
            context_params: Context parameters
            additional_data: Additional method data

        Returns:
            Standardized decomposition result
        """
        try:
            # Validate components
            self._validate_decomposition_components(trend, seasonal, residual)

            # Align component lengths
            trend, seasonal, residual = self._align_component_lengths(
                trend, seasonal, residual
            )

            # Universal component consistency validation
            consistency_result = self._validate_component_consistency(
                trend, seasonal, residual, data
            )

            # Fail-fast approach: return error if consistency validation fails
            if not consistency_result["is_valid"]:
                error_message = f"Component consistency validation failed: {consistency_result['error']:.4f} > {consistency_result['tolerance_used']:.4f}"
                return self.handle_error(
                    ValueError(error_message), "consistency_validation"
                )

            # Calculate reconstruction
            reconstruction = trend + seasonal

            # Calculate quality metrics
            component_strengths = self._calculate_component_strengths(
                trend, seasonal, residual
            )
            reconstruction_error = self._calculate_reconstruction_error(
                data, reconstruction
            )
            quality_score = self._calculate_overall_quality_score(
                component_strengths, reconstruction_error
            )

            # Main result
            result = {
                "trend": trend,
                "seasonal": seasonal,
                "residual": residual,
                "reconstruction": reconstruction,
                "model": self.model_type,
                "component_strengths": component_strengths,
                "quality_score": quality_score,
                "reconstruction_error": reconstruction_error,
                "consistency_validation": consistency_result,
            }

            # Add additional data
            if additional_data:
                result.update(additional_data)

            # Additional metadata for decomposition
            decomposition_metadata = {
                "model_type": self.model_type,
                "confidence_threshold": self.confidence_threshold,
                "main_period": context_params.get("main_period"),
                "component_quality": component_strengths,
                "consistency_error": consistency_result["error"],
                "tolerance_used": consistency_result["tolerance_used"],
                "consistency_warnings": consistency_result["warnings"],
            }

            return self.create_success_response(
                result, data, context_params, decomposition_metadata
            )

        except Exception as e:
            return self.handle_error(e, "decomposition result preparation")

    def _validate_decomposition_components(
        self, trend: pd.Series, seasonal: pd.Series, residual: pd.Series
    ) -> None:
        """
        Validate decomposition components.
        Vectorized version for improved performance.

        Args:
            trend: Trend component
            seasonal: Seasonal component
            residual: Residual component

        Raises:
            ValueError: If components are invalid
        """
        components = {"trend": trend, "seasonal": seasonal, "residual": residual}
        component_names = list(components.keys())
        component_values = list(components.values())

        # Vectorized type checking
        type_checks = np.array(
            [isinstance(comp, pd.Series) for comp in component_values]
        )
        if not np.all(type_checks):
            invalid_indices = np.where(~type_checks)[0]
            for idx in invalid_indices:
                name = component_names[idx]
                comp_type = type(component_values[idx])
                raise ValueError(
                    f"Component '{name}' must be pd.Series, got {comp_type}"
                )

        # Vectorized NaN value checking
        nan_checks = np.array([comp.isna().any() for comp in component_values])
        if np.any(nan_checks):
            nan_indices = np.where(nan_checks)[0]
            for idx in nan_indices:
                name = component_names[idx]
                logging.warning(f"{self} - Component '{name}' contains NaN values")

        # Vectorized infinite value checking
        inf_checks = np.array([np.isinf(comp).any() for comp in component_values])
        if np.any(inf_checks):
            inf_indices = np.where(inf_checks)[0]
            for idx in inf_indices:
                name = component_names[idx]
                raise ValueError(f"Component '{name}' contains infinite values")

    def _align_component_lengths(
        self, trend: pd.Series, seasonal: pd.Series, residual: pd.Series
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Align lengths and indices of decomposition components.

        Args:
            trend: Trend component
            seasonal: Seasonal component
            residual: Residual component

        Returns:
            Tuple of aligned components with consistent indices
        """
        lengths = [len(trend), len(seasonal), len(residual)]
        components = [trend, seasonal, residual]
        names = ["trend", "seasonal", "residual"]

        # Check if components have different lengths or indices
        need_alignment = len(set(lengths)) > 1
        indices_equal = trend.index.equals(seasonal.index) and seasonal.index.equals(
            residual.index
        )

        if need_alignment or not indices_equal:
            logging.warning(
                f"{self} - Components have different lengths or indices, aligning"
            )

            # Find common index intersection
            common_index = trend.index
            for component in [seasonal, residual]:
                common_index = common_index.intersection(component.index)  # type: ignore

            if len(common_index) == 0:
                # Fallback to positional alignment if no common index
                logging.warning(
                    f"{self} - No common index found, using positional alignment"
                )
                min_len = min(lengths)
                trend = trend.iloc[:min_len].reset_index(drop=True)
                seasonal = seasonal.iloc[:min_len].reset_index(drop=True)
                residual = residual.iloc[:min_len].reset_index(drop=True)
            else:
                # Use common index intersection
                trend = trend.loc[common_index]
                seasonal = seasonal.loc[common_index]
                residual = residual.loc[common_index]

                if len(common_index) < max(lengths):
                    lost_points = max(lengths) - len(common_index)
                    logging.warning(
                        f"{self} - Lost {lost_points} data points during index alignment"
                    )

        return trend, seasonal, residual

    def _calculate_reconstruction_error(
        self, original: pd.Series, reconstruction: pd.Series
    ) -> float:
        """
        Calculate reconstruction error.

        Args:
            original: Original time series
            reconstruction: Reconstructed series (trend + seasonal)

        Returns:
            Normalized reconstruction error
        """
        try:
            # Align lengths
            min_len = min(len(original), len(reconstruction))
            orig_aligned = original[:min_len]
            recon_aligned = reconstruction[:min_len]

            # Use universal QualityEvaluator for MAE
            quality_evaluator = QualityEvaluator(evaluation_type="decomposition")
            mae = quality_evaluator.calculate_mae(orig_aligned, recon_aligned)
            std_original = np.std(orig_aligned)

            # Improved handling of edge cases
            if std_original > 0:
                normalized_error = mae / std_original
            elif mae == 0:
                # Perfect reconstruction of constant data
                normalized_error = PERFECT_RECONSTRUCTION_ERROR
            else:
                # Poor reconstruction of constant data - use large penalty
                normalized_error = FAILED_RECONSTRUCTION_PENALTY

            return float(normalized_error)

        except Exception:
            return float("inf")

    def _calculate_component_strengths(
        self, trend: pd.Series, seasonal: pd.Series, residual: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate strength of decomposition components using Hyndman formula.
        Vectorized version for improved performance.

        Args:
            trend: Trend component
            seasonal: Seasonal component
            residual: Residual component

        Returns:
            Dict with strength of each component
        """
        try:
            # Vectorized calculation of variances for all components at once
            components_array = np.array(
                [trend.values, seasonal.values, residual.values]
            )
            component_lengths = np.array([len(trend), len(seasonal), len(residual)])

            # Condition for correct variance calculation (length > 1)
            valid_mask = component_lengths > 1

            # Initialize variance array with zeros
            variances = np.zeros(3)

            # Vectorized variance calculation only for valid components
            if np.any(valid_mask):
                valid_components = components_array[valid_mask]
                variances[valid_mask] = np.var(valid_components, axis=1)

            trend_var, seasonal_var, residual_var = variances
            total_var = np.sum(variances)

            if total_var == 0:
                return {
                    "trend_strength": DEFAULT_TREND_STRENGTH,
                    "seasonal_strength": DEFAULT_SEASONAL_STRENGTH,
                    "residual_strength": DEFAULT_RESIDUAL_STRENGTH,
                }

            # Vectorized calculation of strength values with broadcasting
            strength_values = variances / total_var
            # Apply clipping vectorially
            strength_values = np.clip(strength_values, 0.0, 1.0)

            return {
                "trend_strength": float(strength_values[0]),
                "seasonal_strength": float(strength_values[1]),
                "residual_strength": float(strength_values[2]),
            }

        except Exception:
            logging.warning(f"{self} - Error calculating component strengths")
            return {
                "trend_strength": DEFAULT_TREND_STRENGTH,
                "seasonal_strength": DEFAULT_SEASONAL_STRENGTH,
                "residual_strength": DEFAULT_RESIDUAL_STRENGTH,
            }

    def _calculate_overall_quality_score(
        self, component_strengths: Dict[str, float], reconstruction_error: float
    ) -> float:
        """
        Calculate overall decomposition quality score.

        Args:
            component_strengths: Component strengths
            reconstruction_error: Reconstruction error

        Returns:
            Overall quality score (0-1)
        """
        try:
            trend_strength = component_strengths.get("trend_strength", 0)
            seasonal_strength = component_strengths.get("seasonal_strength", 0)
            residual_strength = component_strengths.get("residual_strength", 1)

            # Decomposition quality is higher when:
            # 1. Trend and seasonality are present
            # 2. Residuals are minimal
            # 3. Reconstruction error is minimal

            signal_strength = trend_strength + seasonal_strength
            noise_penalty = residual_strength
            reconstruction_penalty = min(
                MAX_RECONSTRUCTION_PENALTY, reconstruction_error
            )

            quality_score = (
                signal_strength
                * (ADJUSTMENT_BASE_MULTIPLIER - noise_penalty * NOISE_PENALTY_WEIGHT)
                * (ADJUSTMENT_BASE_MULTIPLIER - reconstruction_penalty)
            )

            return min(
                MAX_QUALITY_SCORE,
                max(MIN_QUALITY_SCORE, quality_score),
            )

        except Exception:
            return FALLBACK_QUALITY_SCORE

    def _vectorized_parameter_adjustment(
        self, parameters: Dict[str, Any], data_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Vectorized parameter adjustment based on data characteristics.

        Args:
            parameters: Base method parameters
            data_characteristics: Time series characteristics

        Returns:
            Adjusted parameters
        """
        adjusted_params = parameters.copy()

        try:
            # Extract data characteristics vectorially
            data_length = data_characteristics.get("length", 0)
            seasonality_strength = data_characteristics.get("seasonality_strength", 0.0)
            trend_strength = data_characteristics.get("trend_strength", 0.0)
            volatility = data_characteristics.get("volatility", 1.0)

            # Vectorized adjustment based on characteristics
            characteristics_array = np.array(
                [
                    seasonality_strength,
                    trend_strength,
                    volatility,
                    min(
                        MAX_QUALITY_SCORE,
                        data_length / DATA_LENGTH_NORMALIZATION_FACTOR,
                    ),
                ]
            )

            # Broadcasting for parameter adjustment
            if "window_size" in adjusted_params:
                # Adaptive window size based on data characteristics
                base_window = adjusted_params["window_size"]
                adjustment_factor = np.dot(
                    characteristics_array, CHARACTERISTIC_WEIGHTS
                )
                adjusted_params["window_size"] = max(
                    MIN_WINDOW_SIZE,
                    int(base_window * (ADJUSTMENT_BASE_MULTIPLIER + adjustment_factor)),
                )

            if "smoothing_level" in adjusted_params:
                # Adaptive smoothing based on volatility
                base_smoothing = adjusted_params["smoothing_level"]
                volatility_adjustment = np.clip(
                    volatility,
                    MIN_VOLATILITY_ADJUSTMENT,
                    MAX_VOLATILITY_ADJUSTMENT,
                )
                adjusted_params["smoothing_level"] = np.clip(
                    base_smoothing / volatility_adjustment,
                    MIN_SMOOTHING_LEVEL,
                    MAX_SMOOTHING_LEVEL,
                )

            if "seasonal_periods" in adjusted_params and isinstance(
                adjusted_params["seasonal_periods"], list
            ):
                # Filter seasonal periods based on data length
                periods_array = np.array(adjusted_params["seasonal_periods"])
                valid_periods = periods_array[
                    periods_array <= data_length // PERIOD_LENGTH_DIVISOR
                ]
                adjusted_params["seasonal_periods"] = (
                    valid_periods.tolist()
                    if len(valid_periods) > 0
                    else [DEFAULT_FALLBACK_PERIOD]
                )

            return adjusted_params

        except Exception as e:
            logging.warning(f"{self} - Error in vectorized parameter adjustment: {e}")
            return parameters

    def create_standard_metadata(
        self,
        data: pd.Series,
        context_params: Dict[str, Any],
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create standard metadata for decomposition result.

        Args:
            data: Time series
            context_params: Parameters from context
            additional_metadata: Additional metadata

        Returns:
            Dict with standard metadata
        """
        # Get base metadata
        metadata = super().create_standard_metadata(
            data, context_params, additional_metadata
        )

        # Add decomposition-specific metadata
        metadata.update(
            {
                "model_type": self.model_type,
                "confidence_threshold": self.confidence_threshold,
                "main_period": context_params.get("main_period"),
            }
        )

        return metadata

    def get_used_parameters(self) -> Dict[str, Any]:
        """
        Get used method parameters.

        Returns:
            Dict with method parameters
        """
        return {
            "name": self.name,
            "model_type": self.model_type,
            "confidence_threshold": self.confidence_threshold,
            "version": __version__,
        }