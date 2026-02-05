"""
Box-Cox Transformation Helper - Enterprise-Grade Universal Implementation

This module provides numerically stable Box-Cox transformation capabilities
for time series processing with automatic scaling, overflow protection,
and comprehensive error handling.

Mathematical Foundation:
- Forward transform: y = (x^λ - 1) / λ for λ ≠ 0, y = ln(x) for λ = 0
- Inverse transform: x = (λy + 1)^(1/λ) for λ ≠ 0, x = exp(y) for λ = 0

Key Features:
- Automatic data scaling for numerical stability
- Chunked processing for memory efficiency
- Multiple fallback strategies for extreme cases
- Support for pd.Series and np.ndarray
- Lambda optimization capabilities
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

__version__ = "1.0.0"

# Box-Cox numerical stability constants
BOX_COX_EPSILON = 1e-8
BOX_COX_MAX_VALUE = 1e10
BOX_COX_MIN_LAMBDA = 1e-6
BOX_COX_SCALE_THRESHOLD = 1e6
BOX_COX_SAFE_EXP_LIMIT = 700

# Default optimization parameters
DEFAULT_LAMBDA_RANGE = (-2.0, 2.0)
DEFAULT_CHUNK_SIZE = 1000


class BoxCoxTransformer:
    """
    Enterprise-grade Box-Cox transformation utility for time series processing.

    Provides numerically stable forward/inverse transformations with automatic
    scaling, overflow protection, and comprehensive error handling.

    Features:
    - Automatic lambda optimization using MLE
    - Intelligent data preprocessing and scaling
    - Chunked processing for large datasets
    - Multiple fallback strategies for numerical stability
    - Support for multiple data formats (Series/ndarray)

    Usage:
        transformer = BoxCoxTransformer()

        # Basic transform with automatic lambda optimization
        transformed_data, lambda_opt = transformer.fit_transform(data)
        original_data = transformer.inverse_transform(transformed_data, lambda_opt)

        # Manual lambda specification
        transformed_data = transformer.transform(data, lambda_value=0.5)
        original_data = transformer.inverse_transform(transformed_data, lambda_value=0.5)
    """

    def __init__(
        self,
        auto_scale: bool = True,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        lambda_range: Tuple[float, float] = DEFAULT_LAMBDA_RANGE,
        verbose: bool = False,
    ):
        """
        Initialize BoxCoxTransformer.

        Args:
            auto_scale: Enable automatic data scaling for numerical stability
            chunk_size: Size of chunks for processing large arrays
            lambda_range: Range for lambda optimization (min_lambda, max_lambda)
            verbose: Enable detailed logging
        """
        self.auto_scale = auto_scale
        self.chunk_size = chunk_size
        self.lambda_range = lambda_range
        self.verbose = verbose

        # Internal state
        self._fitted_lambda = None
        self._scaling_info = None

        # Statistics tracking
        self._transform_stats = {
            "transform_count": 0,
            "inverse_transform_count": 0,
            "optimization_count": 0,
            "fallback_count": 0,
        }

    def __str__(self) -> str:
        """String representation for logging."""
        return f"BoxCoxTransformer(auto_scale={self.auto_scale}, chunk_size={self.chunk_size})"

    def fit(
        self,
        data: Union[pd.Series, np.ndarray],
        lambda_range: Optional[Tuple[float, float]] = None,
    ) -> float:
        """
        Fit transformer to data and find optimal lambda parameter.

        Args:
            data: Input time series data (must be positive)
            lambda_range: Custom range for lambda optimization

        Returns:
            Optimal lambda parameter

        Raises:
            ValueError: If data contains non-positive values without auto_scale
        """
        try:
            # Convert to numpy array for processing
            data_array = self._to_numpy(data)

            # Validate input data
            self._validate_input(data_array)

            # Preprocess data if auto_scale is enabled
            processed_data, scaling_info = self._preprocess_data(data_array)
            self._scaling_info = scaling_info

            # Optimize lambda parameter
            search_range = lambda_range or self.lambda_range
            optimal_lambda = self._optimize_lambda(processed_data, search_range)

            self._fitted_lambda = optimal_lambda
            self._transform_stats["optimization_count"] += 1

            if self.verbose:
                logging.info(
                    f"{self} - Fitted lambda: {optimal_lambda:.6f}, "
                    f"scaling applied: {scaling_info['was_scaled']}"
                )

            return optimal_lambda

        except Exception as e:
            logging.error(f"{self} - Error in fit: {e}")
            raise

    def transform(
        self, data: Union[pd.Series, np.ndarray], lambda_value: Optional[float] = None
    ) -> Union[pd.Series, np.ndarray]:
        """
        Apply Box-Cox transformation to data.

        Args:
            data: Input data to transform
            lambda_value: Lambda parameter (uses fitted lambda if None)

        Returns:
            Transformed data in same format as input

        Raises:
            ValueError: If lambda not provided and transformer not fitted
        """
        try:
            # Use fitted lambda or provided lambda
            lmbda = lambda_value if lambda_value is not None else self._fitted_lambda
            if lmbda is None:
                raise ValueError(
                    "Lambda parameter not provided and transformer not fitted"
                )

            # Convert to numpy for processing
            data_array = self._to_numpy(data)

            # Validate input
            self._validate_input(data_array)

            # Preprocess data
            processed_data, scaling_info = self._preprocess_data(data_array)

            # Apply Box-Cox transformation
            transformed_array = self._forward_transform(processed_data, lmbda)

            self._transform_stats["transform_count"] += 1

            # Return in original format
            return self._to_original_format(transformed_array, data)

        except Exception as e:
            logging.error(f"{self} - Error in transform: {e}")
            raise

    def inverse_transform(
        self,
        data: Union[pd.Series, np.ndarray],
        lambda_value: Optional[float] = None,
        scaling_info: Optional[Dict[str, Any]] = None,
    ) -> Union[pd.Series, np.ndarray]:
        """
        Apply inverse Box-Cox transformation to data.

        Args:
            data: Transformed data to inverse transform
            lambda_value: Lambda parameter used in forward transform
            scaling_info: Scaling information from forward transform

        Returns:
            Original scale data in same format as input
        """
        try:
            # Use fitted lambda or provided lambda
            lmbda = lambda_value if lambda_value is not None else self._fitted_lambda
            if lmbda is None:
                raise ValueError(
                    "Lambda parameter not provided and transformer not fitted"
                )

            # Convert to numpy for processing
            data_array = self._to_numpy(data)

            # Apply inverse transformation
            inverse_array = self._safe_box_cox_inverse(data_array, lmbda)

            # Apply inverse scaling if scaling info provided
            if scaling_info is None:
                scaling_info = self._scaling_info or {"was_scaled": False}

            if scaling_info.get("was_scaled", False):
                inverse_array = self._postprocess_data(inverse_array, scaling_info)

            self._transform_stats["inverse_transform_count"] += 1

            # Return in original format
            return self._to_original_format(inverse_array, data)

        except Exception as e:
            logging.error(f"{self} - Error in inverse_transform: {e}")
            raise

    def fit_transform(
        self,
        data: Union[pd.Series, np.ndarray],
        lambda_range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Union[pd.Series, np.ndarray], float]:
        """
        Fit transformer and apply transformation in one step.

        Args:
            data: Input data
            lambda_range: Range for lambda optimization

        Returns:
            Tuple of (transformed_data, optimal_lambda)
        """
        optimal_lambda = self.fit(data, lambda_range)
        transformed_data = self.transform(data, optimal_lambda)
        return transformed_data, optimal_lambda

    def get_stats(self) -> Dict[str, Any]:
        """Get transformer usage statistics."""
        return self._transform_stats.copy()

    def reset_stats(self) -> None:
        """Reset transformer statistics."""
        for key in self._transform_stats:
            self._transform_stats[key] = 0

    # Private methods for internal functionality

    def _to_numpy(self, data: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Convert input data to numpy array."""
        if isinstance(data, pd.Series):
            return np.asarray(data.values)
        return np.asarray(data)

    def _to_original_format(
        self, array: np.ndarray, original: Union[pd.Series, np.ndarray]
    ) -> Union[pd.Series, np.ndarray]:
        """Convert array back to original data format."""
        if isinstance(original, pd.Series):
            return pd.Series(array, index=original.index, name=original.name)
        return array

    def _validate_input(self, data: np.ndarray) -> None:
        """Validate input data for Box-Cox transformation."""
        if len(data) == 0:
            raise ValueError("Empty data provided")

        if np.any(np.isnan(data)):
            raise ValueError("Data contains NaN values")

        if not self.auto_scale and np.any(data <= 0):
            raise ValueError(
                "Data contains non-positive values. Enable auto_scale or "
                "preprocess data to ensure all values are positive."
            )

    def _preprocess_data(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocess data for numerical stability.

        Args:
            data: Input data array

        Returns:
            Tuple of (processed_data, scaling_info)
        """
        scaling_info = {
            "scale_factor": 1.0,
            "offset": 0.0,
            "was_scaled": False,
            "original_range": (float(np.min(data)), float(np.max(data))),
        }

        if not self.auto_scale:
            return data, scaling_info

        data_values = data.copy()

        # Check if scaling is needed
        needs_scaling = (
            np.max(np.abs(data_values)) > BOX_COX_SCALE_THRESHOLD
            or np.min(data_values) <= 0
        )

        if needs_scaling:
            if self.verbose:
                logging.info(
                    f"{self} - Applying data scaling for numerical stability. "
                    f"Original range: [{np.min(data_values):.3e}, {np.max(data_values):.3e}]"
                )

            # Handle negative/zero values
            if np.min(data_values) <= 0:
                offset = abs(np.min(data_values)) + 1.0
                data_values = data_values + offset
                scaling_info["offset"] = offset
                if self.verbose:
                    logging.info(
                        f"{self} - Applied offset {offset:.3f} for positive values"
                    )

            # Scale large values
            if np.max(data_values) > BOX_COX_SCALE_THRESHOLD:
                scale_factor = BOX_COX_SCALE_THRESHOLD / np.max(data_values)
                data_values = data_values * scale_factor
                scaling_info["scale_factor"] = scale_factor
                if self.verbose:
                    logging.info(f"{self} - Applied scaling factor {scale_factor:.3e}")

            scaling_info["was_scaled"] = True

            if self.verbose:
                logging.info(
                    f"{self} - Scaled range: [{np.min(data_values):.3e}, {np.max(data_values):.3e}]"
                )

        return data_values, scaling_info

    def _postprocess_data(
        self, data: np.ndarray, scaling_info: Dict[str, Any]
    ) -> np.ndarray:
        """
        Reverse scaling applied in preprocessing.

        Args:
            data: Processed data array
            scaling_info: Scaling information from preprocessing

        Returns:
            Data in original scale
        """
        if not scaling_info.get("was_scaled", False):
            return data

        result = data.copy()
        scale_factor = scaling_info["scale_factor"]
        offset = scaling_info["offset"]

        if self.verbose:
            logging.info(
                f"{self} - Reversing scaling: factor={scale_factor:.3e}, offset={offset:.3f}"
            )

        # Reverse scaling
        if scale_factor != 1.0:
            result = result / scale_factor

        # Reverse offset
        if offset != 0.0:
            result = result - offset

        return result

    def _optimize_lambda(
        self, data: np.ndarray, lambda_range: Tuple[float, float]
    ) -> float:
        """
        Optimize lambda parameter using maximum likelihood estimation.

        Args:
            data: Preprocessed positive data
            lambda_range: Range for lambda search

        Returns:
            Optimal lambda parameter
        """
        try:
            # Use scipy's boxcox for optimization
            boxcox_result = stats.boxcox(data, lmbda=None)
            optimal_lambda = boxcox_result[1]  # Extract lambda from tuple

            # Constrain lambda to specified range
            min_lambda, max_lambda = lambda_range
            optimal_lambda = float(np.clip(optimal_lambda, min_lambda, max_lambda))

            return optimal_lambda

        except Exception as e:
            if self.verbose:
                logging.warning(
                    f"{self} - Lambda optimization failed: {e}. Using lambda=0.0"
                )
            return 0.0  # Default to log transformation

    def _forward_transform(self, data: np.ndarray, lmbda: float) -> np.ndarray:
        """
        Apply forward Box-Cox transformation.

        Args:
            data: Input data (must be positive)
            lmbda: Lambda parameter

        Returns:
            Transformed data
        """
        if abs(lmbda) < BOX_COX_MIN_LAMBDA:
            # Lambda ≈ 0: log transformation
            return np.log(np.maximum(data, BOX_COX_EPSILON))
        else:
            # Standard Box-Cox transformation
            with np.errstate(over="ignore", invalid="ignore"):
                try:
                    result = (np.power(data, lmbda) - 1) / lmbda

                    # Check for overflow/invalid results
                    if np.any(~np.isfinite(result)):
                        if self.verbose:
                            logging.warning(
                                f"{self} - Forward transform overflow, using log fallback"
                            )
                        return np.log(np.maximum(data, BOX_COX_EPSILON))

                    return result

                except (FloatingPointError, RuntimeWarning):
                    if self.verbose:
                        logging.warning(
                            f"{self} - Forward transform error, using log fallback"
                        )
                    return np.log(np.maximum(data, BOX_COX_EPSILON))

    def _safe_box_cox_inverse(
        self, values: np.ndarray, lmbda: float, scale_factor: float = 1.0
    ) -> np.ndarray:
        """
        Numerically stable Box-Cox inverse transformation.

        Args:
            values: Transformed values
            lmbda: Box-Cox lambda parameter
            scale_factor: Scaling factor to reverse pre-scaling

        Returns:
            Values in original scale
        """
        values = np.asarray(values)

        if abs(lmbda) < BOX_COX_MIN_LAMBDA:
            # Lambda ≈ 0: log transformation case
            # Clip to prevent overflow in exp
            clipped_values = np.clip(
                values, -BOX_COX_SAFE_EXP_LIMIT, BOX_COX_SAFE_EXP_LIMIT
            )
            result = np.exp(clipped_values)
        else:
            # Standard Box-Cox case: y = (x^lambda - 1) / lambda
            # Inverse: x = (lambda * y + 1)^(1/lambda)

            # Protect base values
            base = 1.0 + lmbda * values
            base = np.clip(base, BOX_COX_EPSILON, BOX_COX_MAX_VALUE)

            exponent = 1.0 / lmbda

            # Check for extreme exponents
            if abs(exponent) > 50:
                if self.verbose:
                    logging.warning(
                        f"Extreme Box-Cox exponent {exponent:.3f}, using log approximation"
                    )
                # Use log approximation for numerical stability
                result = np.exp(np.log(base) / lmbda)
            else:
                # Use chunked computation for large arrays to avoid memory issues
                if len(values) > self.chunk_size:
                    result = self._chunked_power(base, exponent)
                else:
                    with np.errstate(over="raise", invalid="raise"):
                        try:
                            result = np.power(base, exponent)
                        except (FloatingPointError, RuntimeWarning):
                            if self.verbose:
                                logging.warning(
                                    "Box-Cox power overflow, using log approximation"
                                )
                            result = np.exp(np.log(base) / lmbda)
                            self._transform_stats["fallback_count"] += 1

        # Apply scale factor to reverse pre-scaling
        result = result * scale_factor

        # Final sanity check
        if np.any(~np.isfinite(result)):
            if self.verbose:
                logging.warning(
                    "Non-finite values in Box-Cox inverse, applying clipping"
                )
            result = np.clip(result, BOX_COX_EPSILON, BOX_COX_MAX_VALUE)

        return result

    def _chunked_power(self, base: np.ndarray, exponent: float) -> np.ndarray:
        """
        Compute power operation in chunks to avoid memory overflow.

        Args:
            base: Base values
            exponent: Power exponent

        Returns:
            Power result
        """
        result = np.zeros_like(base)

        for i in range(0, len(base), self.chunk_size):
            end_idx = min(i + self.chunk_size, len(base))
            chunk = base[i:end_idx]

            with np.errstate(over="ignore", invalid="ignore"):
                chunk_result = np.power(chunk, exponent)

                # Check for overflow in chunk
                if np.any(~np.isfinite(chunk_result)):
                    # Fallback to log computation for this chunk
                    chunk_result = np.exp(np.log(chunk) / (1.0 / exponent))
                    self._transform_stats["fallback_count"] += 1

                result[i:end_idx] = chunk_result

        return result


# Convenience functions for quick usage


def boxcox_transform(
    data: Union[pd.Series, np.ndarray],
    lambda_value: Optional[float] = None,
    auto_scale: bool = True,
) -> Tuple[Union[pd.Series, np.ndarray], float]:
    """
    Quick Box-Cox transformation with automatic lambda optimization.

    Args:
        data: Input data
        lambda_value: Lambda parameter (auto-optimized if None)
        auto_scale: Enable automatic scaling for numerical stability

    Returns:
        Tuple of (transformed_data, lambda_used)
    """
    transformer = BoxCoxTransformer(auto_scale=auto_scale)

    if lambda_value is None:
        return transformer.fit_transform(data)
    else:
        transformed = transformer.transform(data, lambda_value)
        return transformed, lambda_value


def boxcox_inverse_transform(
    data: Union[pd.Series, np.ndarray],
    lambda_value: float,
    scaling_info: Optional[Dict[str, Any]] = None,
) -> Union[pd.Series, np.ndarray]:
    """
    Quick Box-Cox inverse transformation.

    Args:
        data: Transformed data
        lambda_value: Lambda parameter used in forward transform
        scaling_info: Scaling information from forward transform

    Returns:
        Original scale data
    """
    transformer = BoxCoxTransformer()
    return transformer.inverse_transform(data, lambda_value, scaling_info)
