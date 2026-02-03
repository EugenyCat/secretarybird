"""
SSA (Singular Spectrum Analysis) time series decomposition method.

SSA uses SVD decomposition of the trajectory matrix to extract trend
and periodic components without stationarity assumptions.
Optimal for non-stationary data with high autocorrelation.

Mathematical foundations:
- Hankel trajectory matrix of size L×K
- SVD decomposition to extract principal components
- Eigenvalue-based component grouping
- Diagonal averaging for time series reconstruction

Application conditions (Enhanced Decision Tree):
- NOT is_stationary AND lag1_autocorr > 0.95
- Alternatively: NOT is_stationary AND noise > 0.2 AND data_quality < 0.7
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided
from scipy import linalg

from pipeline.helpers.utils import validate_required_locals
from pipeline.timeSeriesProcessing.decomposition.methods.baseDecomposerMethod import (
    BaseDecomposerMethod,
)

__version__ = "1.1.0"

# SSA-specific mathematical constants
class SSAConstants:
    """Mathematical constants for SSA (Singular Spectrum Analysis) decomposition."""

    # Validation constants
    MIN_VARIANCE_THRESHOLD = 0.1  # Minimum acceptable variance threshold
    MAX_VARIANCE_THRESHOLD = 1.0  # Maximum acceptable variance threshold

    # SVD and matrix processing constants
    TRUNCATED_SVD_SIZE_THRESHOLD = 100  # Matrix size threshold for truncated SVD
    MAX_EIGENVALUES_LOG = 10  # Maximum eigenvalues to log for diagnostics
    DEFAULT_DATA_TYPE = np.float64  # Numerical precision for SSA calculations

    # Data preprocessing constants
    DEFAULT_MEAN_FALLBACK = 0.0  # Fallback mean when normalization fails
    DEFAULT_STD_FALLBACK = 1.0  # Fallback std when normalization fails
    ZERO_COUNT_REPLACEMENT = 1.0  # Replacement for zero counts in diagonal averaging

    # Component grouping thresholds
    COMPONENT_GROUP_THRESHOLDS = [2, 5, 10]  # Thresholds for automatic component grouping

    # Trend component ratios for different data sizes
    MIN_TREND_COMPONENTS = 1  # Minimum trend components
    SMALL_DATA_TREND_COMPONENTS = 2  # Trend components for small datasets
    MEDIUM_DATA_TREND_DIVISOR = 2  # Divisor for medium data: n_components // 2
    LARGE_DATA_TREND_DIVISOR = 3  # Divisor for large data: n_components // 3
    MAX_TREND_COMPONENTS_SMALL = 3  # Max trend components for small data
    MAX_TREND_COMPONENTS_LARGE = 5  # Max trend components for large data

    # Early stopping constants
    DEFAULT_VARIANCE_THRESHOLD_EARLY_STOP = 0.95  # Threshold for early SVD termination
    MIN_SIGNIFICANT_COMPONENTS = 3  # Minimum components before early stopping

    # Fallback values
    FALLBACK_VARIANCE_EXPLAINED = 0.0  # Fallback when variance calculation fails


class SSADecomposerMethod(BaseDecomposerMethod):
    """
    SSA (Singular Spectrum Analysis) decomposition method.

    Uses SVD decomposition of trajectory matrix to extract components
    of time series without assumptions about periodicity or stationarity.
    """

    DEFAULT_CONFIG = {
        **BaseDecomposerMethod.DEFAULT_CONFIG,
        # Algorithmic constants
        "component_grouping": "automatic",  # automatic is always better than manual (Does not require adaptation)
        # Main SSA trajectory matrix parameters (adapted in configDecomposition)
        # "window_length": None,  # [AUTO] data_length/4 optimal for long-term dependencies
        # "n_components": None,  # [AUTO] based on variance_threshold and max_components
        # SVD parameters (adapted in configDecomposition)
        # "variance_threshold": 0.85,  # [AUTO] may adapt based on data quality (0.80-0.90)
        # "max_components": 50,  # [AUTO] may adapt based on data size
        # "svd_method": "truncated",  # [AUTO] full for small data, truncated for large
        # "normalize": True,  # [AUTO] adapted based on volatility (volatility > 0.5)
        # TODO not urgent: Improve using these parameters (not used in current implementation):
        # "trend_ratio": 0.3 - exists in configDecomposition, but NOT used in code (inactive)
        # "min_variance_explained": 0.5 - exists in configDecomposition, but NOT used in code (inactive)
        # "confidence_threshold": 0.80 - may be inherited from BaseDecomposerMethod
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize SSA method."""
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged_config)

        # Validate SSA-specific parameters
        validate_required_locals(
            ["window_length", "variance_threshold", "max_components"], self.config
        )

        # Extract parameters for quick access
        self.window_length = self.config["window_length"]
        self.variance_threshold = self.config["variance_threshold"]
        self.max_components = self.config["max_components"]
        self.svd_method = self.config["svd_method"]
        self.normalize = self.config["normalize"]

    def __str__(self) -> str:
        """String representation for logging."""
        return (
            f"SSADecomposerMethod(window={self.window_length}, "
            f"variance_threshold={self.variance_threshold}, "
            f"max_components={self.max_components})"
        )

    def _validate_config(self) -> None:
        """Validate SSA method configuration."""
        # Check window_length
        if (
            self.config["window_length"] is not None
            and self.config["window_length"] <= 1
        ):
            raise ValueError(
                f"window_length must be > 1, got {self.config['window_length']}"
            )

        # Check variance_threshold
        if not SSAConstants.MIN_VARIANCE_THRESHOLD <= self.config["variance_threshold"] <= SSAConstants.MAX_VARIANCE_THRESHOLD:
            raise ValueError(
                f"variance_threshold must be in range [{SSAConstants.MIN_VARIANCE_THRESHOLD}, {SSAConstants.MAX_VARIANCE_THRESHOLD}], "
                f"got {self.config['variance_threshold']}"
            )

        # Check max_components
        if self.config["max_components"] <= 0:
            raise ValueError(
                f"max_components must be > 0, got {self.config['max_components']}"
            )

    def process(
        self, data: pd.Series, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform SSA decomposition of time series.

        Args:
            data: Time series for decomposition
            context: Context with additional information

        Returns:
            Standardized decomposition result
        """
        try:
            # 1. Validate input data
            validation = self.validate_input(data)
            if validation["status"] == "error":
                return validation

            # 2. Extract context (base method)
            context_params = self.extract_context_parameters(context)

            # 3. Preprocess data (base method)
            processed_data = self.preprocess_data(data)

            # 4. Validate and adapt window_length
            window_length = self._validate_window_length(len(processed_data))
            if window_length is None:
                return self.handle_error(
                    ValueError(
                        f"SSA impossible: data too short ({len(processed_data)} points) "
                        f"for window_length={self.window_length}"
                    ),
                    "window length validation",
                )

            logging.info(
                f"{self} - Starting SSA decomposition: length={len(processed_data)}, "
                f"window_length={window_length}"
            )

            # 5. ONLY SSA algorithmic logic
            decomposition_result = self._perform_ssa_decomposition(
                processed_data, window_length
            )

            # 6. Prepare additional data
            additional_data = {
                "window_length": window_length,
                "n_components_used": decomposition_result["n_components_used"],
                "variance_explained": decomposition_result["variance_explained"],
                "eigenvalues": decomposition_result["eigenvalues"][:SSAConstants.MAX_EIGENVALUES_LOG],
                "component_grouping": decomposition_result["component_grouping"],
                "svd_method": self.svd_method,
                "algorithm": "SSA (Singular Spectrum Analysis)",
                "model_type": "nonparametric",
            }

            # 7. Result through base method
            return self.prepare_decomposition_result(
                trend=decomposition_result["trend"],
                seasonal=decomposition_result["seasonal"],
                residual=decomposition_result["residual"],
                data=data,
                context_params=context_params,
                additional_data=additional_data,
            )

        except Exception as e:
            return self.handle_error(e, "SSA decomposition")

    def _validate_window_length(self, data_length: int) -> Optional[int]:
        """
        Validate and adapt window_length for data.

        Args:
            data_length: Time series length

        Returns:
            Validated window length or None if impossible
        """
        if self.window_length is None:
            # Automatic determination: data_length/4 optimal for long-term dependencies
            window_length = max(2, data_length // 4)
        else:
            window_length = self.window_length

        # Critical SSA constraints
        min_window = 2
        max_window = data_length - 1

        if window_length < min_window:
            logging.warning(
                f"{self} - window_length {window_length} too small, using {min_window}"
            )
            window_length = min_window
        elif window_length >= max_window:
            logging.warning(
                f"{self} - window_length {window_length} too large, using {max_window}"
            )
            window_length = max_window

        # Check SSA feasibility
        if window_length >= data_length or data_length < 4:
            logging.error(
                f"{self} - SSA impossible: window_length={window_length} >= data_length={data_length}"
            )
            return None

        return window_length

    def _perform_ssa_decomposition(
        self, data: pd.Series, window_length: int
    ) -> Dict[str, Any]:
        """
        Perform complete SSA decomposition.

        Args:
            data: Preprocessed time series
            window_length: Window length for trajectory matrix

        Returns:
            Dict with decomposition components and metadata
        """
        # 1. Normalize data (optional)
        data_values = np.asarray(data.values, dtype=SSAConstants.DEFAULT_DATA_TYPE)
        if self.normalize:
            data_mean = float(np.mean(data_values))
            data_std = float(np.std(data_values))
            if data_std > 0:
                data_values = (data_values - data_mean) / data_std
            else:
                data_mean, data_std = SSAConstants.DEFAULT_MEAN_FALLBACK, SSAConstants.DEFAULT_STD_FALLBACK

        # 2. Create Hankel trajectory matrix
        trajectory_matrix = self._create_trajectory_matrix(data_values, window_length)

        # 3. SVD decomposition
        U, sigma, Vt = self._perform_svd(trajectory_matrix)

        # 4. Determine number of significant components
        n_components = self._determine_n_components(sigma)

        # 5. Group components into trend and seasonality
        trend_indices, seasonal_indices = self._group_components(
            U, sigma, Vt, n_components, len(data_values)
        )

        # 6. Reconstruct components
        trend_values = self._reconstruct_component(
            U, sigma, Vt, trend_indices, len(data_values)
        )
        seasonal_values = self._reconstruct_component(
            U, sigma, Vt, seasonal_indices, len(data_values)
        )

        # 7. Denormalize (if normalization was applied)
        if self.normalize and data_std > 0:
            trend_values = trend_values * data_std + data_mean
            seasonal_values = seasonal_values * data_std

        # 8. Create pandas Series with correct indices
        trend = pd.Series(trend_values[: len(data)], index=data.index)
        seasonal = pd.Series(seasonal_values[: len(data)], index=data.index)
        residual = data - trend - seasonal

        # 9. Decomposition metadata
        variance_explained = self._calculate_variance_explained(
            sigma, trend_indices + seasonal_indices
        )

        return {
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
            "n_components_used": len(trend_indices) + len(seasonal_indices),
            "variance_explained": variance_explained,
            "eigenvalues": sigma.tolist(),
            "component_grouping": {
                "trend_components": trend_indices,
                "seasonal_components": seasonal_indices,
                "total_components": n_components,
            },
        }

    def _create_trajectory_matrix(
        self, data: np.ndarray, window_length: int
    ) -> np.ndarray:
        """
        Create Hankel trajectory matrix efficiently.

        Args:
            data: Time series
            window_length: Window length

        Returns:
            Trajectory matrix of size (window_length, K)
        """
        N = len(data)
        K = N - window_length + 1

        # Efficient creation via stride_tricks
        shape = (window_length, K)
        strides = (data.strides[0], data.strides[0])

        # Create matrix with copy for safety
        trajectory_matrix = as_strided(data, shape=shape, strides=strides).copy()

        logging.debug(
            f"{self} - Trajectory matrix created: {trajectory_matrix.shape}"
        )
        return trajectory_matrix

    def _perform_svd(
        self, matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform SVD decomposition with optimal method selection.

        Args:
            matrix: Trajectory matrix

        Returns:
            Tuple (U, sigma, Vt) - SVD result
        """
        try:
            if self.svd_method == "truncated" and min(matrix.shape) > SSAConstants.TRUNCATED_SVD_SIZE_THRESHOLD:
                # For large matrices use truncated SVD
                # Limit number of components for efficiency
                k = min(self.max_components, min(matrix.shape) - 1)
                U, sigma, Vt = linalg.svd(matrix, full_matrices=False)

                # Early stopping: check cumulative variance for early termination
                k_early_stop = self._apply_early_stopping_svd(sigma, k)
                if k_early_stop < k:
                    U, sigma, Vt = U[:, :k_early_stop], sigma[:k_early_stop], Vt[:k_early_stop, :]
                    logging.info(f"{self} - Early stopping applied: reduced from k={k} to k={k_early_stop}")
                else:
                    U, sigma, Vt = U[:, :k], sigma[:k], Vt[:k, :]
                logging.debug(f"{self} - Truncated SVD performed: k={len(sigma)}")
            else:
                # Full SVD for small data
                U, sigma, Vt = linalg.svd(matrix, full_matrices=False)

                # Early stopping for full SVD as well
                k_early_stop = self._apply_early_stopping_svd(sigma, len(sigma))
                if k_early_stop < len(sigma):
                    U, sigma, Vt = U[:, :k_early_stop], sigma[:k_early_stop], Vt[:k_early_stop, :]
                    logging.info(f"{self} - Early stopping applied: reduced from {len(sigma)} to k={k_early_stop} components")

                logging.debug(f"{self} - Full SVD performed: final components={len(sigma)}")

            return U, sigma, Vt

        except linalg.LinAlgError as e:
            raise ValueError(f"SVD decomposition failed: {str(e)}")

    def _apply_early_stopping_svd(self, sigma: np.ndarray, max_k: int) -> int:
        """
        Apply early stopping for SVD based on cumulative variance.

        Stops adding components when high variance threshold is reached
        or when number of significant components reaches minimum.

        Args:
            sigma: Singular values (sorted in descending order)
            max_k: Maximum number of components to consider

        Returns:
            Optimal number of components for early stopping
        """
        if len(sigma) < SSAConstants.MIN_SIGNIFICANT_COMPONENTS:
            return len(sigma)

        # Calculate cumulative explained variance
        total_variance = np.sum(sigma**2)
        if total_variance == 0:
            return SSAConstants.MIN_SIGNIFICANT_COMPONENTS

        cumulative_variance = np.cumsum(sigma**2) / total_variance

        # Find point where early stopping threshold is reached
        early_stop_idx = np.searchsorted(
            cumulative_variance,
            SSAConstants.DEFAULT_VARIANCE_THRESHOLD_EARLY_STOP
        )

        # Apply minimum constraints
        optimal_k = max(
            SSAConstants.MIN_SIGNIFICANT_COMPONENTS,
            min(early_stop_idx + 1, max_k, len(sigma))
        )

        # Logging for diagnostics
        if optimal_k < max_k:
            variance_at_stop = cumulative_variance[optimal_k - 1] if optimal_k > 0 else 0
            logging.debug(
                f"{self} - Early stopping at {optimal_k} components "
                f"(variance explained: {variance_at_stop:.3f}, threshold: {SSAConstants.DEFAULT_VARIANCE_THRESHOLD_EARLY_STOP})"
            )

        return optimal_k

    def _determine_n_components(self, sigma: np.ndarray) -> int:
        """
        Determine number of significant components by variance_threshold.

        Args:
            sigma: Singular values

        Returns:
            Number of components to use
        """
        # Cumulative explained variance
        total_variance = np.sum(sigma**2)
        explained_variance_ratio = np.cumsum(sigma**2) / total_variance

        # Find number of components to reach threshold
        n_components = (
            np.searchsorted(explained_variance_ratio, self.variance_threshold) + 1
        )

        # Apply constraints
        n_components = min(n_components, self.max_components, len(sigma))
        n_components = max(1, n_components)

        explained_ratio = (
            explained_variance_ratio[n_components - 1] if n_components > 0 else 0
        )
        logging.info(
            f"{self} - Selected {n_components} components, "
            f"explaining {explained_ratio:.1%} of variance"
        )

        return n_components

    def _group_components(
        self,
        U: np.ndarray,
        sigma: np.ndarray,
        Vt: np.ndarray,
        n_components: int,
        data_length: int,
    ) -> Tuple[List[int], List[int]]:
        """
        Intelligent grouping of components into trend and seasonality.
        Vectorized version for improved performance.

        Args:
            U: Left singular vectors
            sigma: Singular values
            Vt: Right singular vectors
            n_components: Number of components to group
            data_length: Original data length

        Returns:
            Tuple (trend_indices, seasonal_indices)
        """
        # Vectorized heuristic for component grouping

        # Determine number of trend components vectorized
        # Use numpy conditional operations instead of if-elif
        thresholds = np.array(SSAConstants.COMPONENT_GROUP_THRESHOLDS)
        n_trend_options = np.array(
            [
                SSAConstants.MIN_TREND_COMPONENTS,
                SSAConstants.SMALL_DATA_TREND_COMPONENTS,
                min(SSAConstants.MAX_TREND_COMPONENTS_SMALL, n_components // SSAConstants.MEDIUM_DATA_TREND_DIVISOR),
                min(SSAConstants.MAX_TREND_COMPONENTS_LARGE, n_components // SSAConstants.LARGE_DATA_TREND_DIVISOR)
            ]
        )

        # Vectorized determination of number of trend components
        condition_index = np.searchsorted(thresholds, n_components, side="right")
        n_trend = n_trend_options[condition_index]

        # Generate lists vectorized
        trend_components = list(np.arange(n_trend))
        seasonal_components = list(np.arange(n_trend, n_components))

        # 2. Can add more complex logic based on:
        # - Analysis of frequency characteristics of components
        # - Autocorrelation analysis of reconstructed series
        # - Spectral analysis of right singular vectors

        logging.debug(
            f"{self} - Component grouping: trend={trend_components}, "
            f"seasonal={seasonal_components}"
        )

        return trend_components, seasonal_components

    def _reconstruct_component(
        self,
        U: np.ndarray,
        sigma: np.ndarray,
        Vt: np.ndarray,
        component_indices: List[int],
        target_length: int,
    ) -> np.ndarray:
        """
        Reconstruct component from selected indices with diagonal averaging.

        Args:
            U: Left singular vectors
            sigma: Singular values
            Vt: Right singular vectors
            component_indices: Component indices for reconstruction
            target_length: Target length of output series

        Returns:
            Reconstructed time series
        """
        if not component_indices:
            return np.zeros(target_length)

        # Reconstruct selected components
        X_reconstructed = np.zeros((U.shape[0], Vt.shape[1]))

        for i in component_indices:
            if i < len(sigma):
                X_reconstructed += sigma[i] * np.outer(U[:, i], Vt[i, :])

        # Diagonal averaging (Hankelization) to get time series
        reconstructed = self._diagonal_averaging(X_reconstructed)

        # Truncate to needed length
        return reconstructed[:target_length]

    def _diagonal_averaging(self, matrix: np.ndarray) -> np.ndarray:
        """
        Diagonal averaging to transform matrix to time series.
        Vectorized version for improved performance.

        Args:
            matrix: Matrix for averaging

        Returns:
            Time series
        """
        window_length, K = matrix.shape
        N = window_length + K - 1
        reconstructed = np.zeros(N)
        counts = np.zeros(N)

        # Vectorized averaging over antidiagonals
        # Create index grid
        i_indices, j_indices = np.meshgrid(
            np.arange(window_length), np.arange(K), indexing="ij"
        )
        # Calculate target indices for antidiagonal averaging
        target_indices = i_indices + j_indices

        # Flat matrix for vectorization
        flat_matrix = matrix.flatten()
        flat_targets = target_indices.flatten()

        # Use np.bincount for efficient summation
        reconstructed = np.bincount(flat_targets, weights=flat_matrix, minlength=N)
        counts = np.bincount(flat_targets, minlength=N).astype(np.float64)

        # Normalization (avoid division by zero)
        counts[counts == 0] = SSAConstants.ZERO_COUNT_REPLACEMENT
        reconstructed = reconstructed / counts

        return reconstructed

    def _calculate_variance_explained(
        self, sigma: np.ndarray, component_indices: List[int]
    ) -> float:
        """
        Calculate fraction of variance explained by selected components.

        Args:
            sigma: Singular values
            component_indices: Selected component indices

        Returns:
            Fraction of explained variance [0, 1]
        """
        if not component_indices or len(sigma) == 0:
            return SSAConstants.FALLBACK_VARIANCE_EXPLAINED

        total_variance = np.sum(sigma**2)
        selected_variance = sum(
            sigma[i] ** 2 for i in component_indices if i < len(sigma)
        )

        return float(selected_variance / total_variance) if total_variance > 0 else SSAConstants.FALLBACK_VARIANCE_EXPLAINED