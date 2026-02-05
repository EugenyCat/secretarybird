"""
MSTL method for time series decomposition - REFACTORED VERSION.

Multiple Seasonal-Trend decomposition for time series with multiple seasonality.
Mathematically correct implementation with minimal overhead.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL

from pipeline.helpers.utils import validate_required_locals
from pipeline.timeSeriesProcessing.decomposition.methods.baseDecomposerMethod import (
    BaseDecomposerMethod,
)

__version__ = "2.0.0"


# Validation constants
MIN_PERIODS_REQUIRED = 2  # MSTL requires at least 2 seasonal periods
MIN_PERIOD_VALUE = 1  # All periods must be > 1

# STL component defaults
DEFAULT_LOW_PASS_DEGREE = 1  # Default degree for low-pass filter

# Component extraction constants
SINGLE_COLUMN_INDEX = 0  # Index for single seasonal component

# Performance monitoring
PERFORMANCE_LOG_THRESHOLD = 1.0  # Log execution time if > 1 second


class MSTLDecomposerMethod(BaseDecomposerMethod):
    """
    MSTL decomposition method for time series with multiple seasonality.

    MSTL (Multiple Seasonal-Trend decomposition) extends classical STL
    to work with multiple seasonal periods simultaneously.

    Architectural refactoring v2.0.0:
    - ONLY mathematical MSTL logic (200-300 lines)
    - Full use of configDecomposition.py for parameter adaptation
    - Compliance with SOLID, KISS, DRY architectural standards
    """

    # Minimal default configurations for MSTL
    DEFAULT_CONFIG = {
        **BaseDecomposerMethod.DEFAULT_CONFIG,
        # Constants
        "robust": True,  # Robust regression for outliers (Does not require adaptation)
        "preprocessing_enabled": True,  # Data preprocessing enabled (Does not require adaptation)
        # Main MSTL parameters
        # "periods": [],            # Multiple seasonality periods (adapted in configDecomposition)
        # "windows": None,          # Window sizes for each period (adapted in configDecomposition)
        # "iterate": 2,             # Number of MSTL algorithm iterations (adapted in configDecomposition)
        # "lmbda": None,            # Box-Cox transformation parameter (adapted in configDecomposition)
        # "seasonal_deg": 1,        # Seasonal component polynomial degree (adapted in configDecomposition)
        # "trend_deg": 1,           # Trend component polynomial degree (adapted in configDecomposition)
        # "inner_iter": None,       # STL inner iterations (adapted in configDecomposition)
        # "outer_iter": None,       # STL outer iterations (adapted in configDecomposition)
        # "trend": None,            # Trend window size (adapted in configDecomposition)
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize MSTL method.

        Args:
            config: Fully adapted configuration from configDecomposition.py
        """
        # Configuration MUST be pre-adapted through configDecomposition.py
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged_config)

        # Validate ONLY MSTL-specific parameters (from configDecomposition)
        validate_required_locals(
            [
                "periods",
                "windows",
                "iterate",
                "trend",
                "seasonal_deg",
                "trend_deg",
                "robust",
                "inner_iter",
                "outer_iter",
            ],
            self.config,
        )

    def __str__(self) -> str:
        """Standard string representation for MSTL logging."""
        periods = self.config.get("periods", [])
        return (
            f"MSTLDecomposerMethod(v{__version__}, periods={len(periods)}, "
            f"iterate={self.config['iterate']}, robust={self.config['robust']})"
        )

    def _validate_config(self) -> None:
        """
        Validate MSTL method configuration.
        MINIMAL validation - main adaptation in configDecomposition.py.
        Vectorized version for improved performance.
        """
        # Critical validation of periods (already adapted in configDecomposition)
        periods = self.config["periods"]
        if (
            not isinstance(periods, list)
            or len(periods) < MIN_PERIODS_REQUIRED
        ):
            raise ValueError(
                f"MSTL periods must be a list with at least {MIN_PERIODS_REQUIRED} elements, got {periods}"
            )

        # Vectorized validation of all periods at once
        periods_array = np.array(periods)

        # Check types: all must be numeric
        if not np.issubdtype(periods_array.dtype, np.number):
            # Fallback for mixed types - check each element
            for i, period in enumerate(periods):
                if not isinstance(period, (int, float)):
                    raise ValueError(
                        f"MSTL period at index {i} must be numeric, got {type(period).__name__}: {period}"
                    )

        # Vectorized check that all periods > MIN_PERIOD_VALUE
        invalid_periods = periods_array <= MIN_PERIOD_VALUE
        if np.any(invalid_periods):
            invalid_indices = np.where(invalid_periods)[0]
            invalid_values = periods_array[invalid_indices]
            raise ValueError(
                f"MSTL periods must be > {MIN_PERIOD_VALUE}, got invalid values at indices {invalid_indices.tolist()}: {invalid_values.tolist()}"
            )

    def process(
        self, data: pd.Series, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform MSTL decomposition.

        ARCHITECTURAL PATTERN v2.0.0: validation → context → preprocessing → algorithm → result

        Args:
            data: Time series for decomposition
            context: Processing context (used through base methods)

        Returns:
            Standardized decomposition result
        """
        try:
            # 1. CRITICAL fail-fast validation
            critical_validation = self.validate_input_critical(data)
            if critical_validation is not None:
                return critical_validation

            # 2. Standard input validation (base method)
            validation = self.validate_input(data)
            if validation["status"] == "error":
                return validation

            # 2. Extract context (base method)
            context_params = self.extract_context_parameters(context)

            # 3. Preprocessing (base method)
            processed_data = self.preprocess_data(data)

            logging.info(
                f"{self} - Starting MSTL decomposition: length={len(processed_data)}, "
                f"periods={self.config['periods']}"
            )

            # 4. ONLY MSTL algorithmic logic
            # DO NOT extract periods/parameters from context directly
            # Use self.config (already adapted in configDecomposition.py)
            mstl_result = self._perform_mstl_decomposition(processed_data)

            # 5. Extract MSTL components
            trend, seasonal_combined, residual, additional_data = (
                self._extract_mstl_components(mstl_result)
            )

            # 6. Result through base method
            result = self.prepare_decomposition_result(
                trend=trend,
                seasonal=seasonal_combined,
                residual=residual,
                data=processed_data,
                context_params=context_params,
                additional_data=additional_data,
            )

            logging.info(
                f"{self} - MSTL decomposition completed with {len(self.config['periods'])} periods"
            )

            return result

        except Exception as e:
            # Use base class utility method for error handling
            return self.handle_error(e, "MSTL decomposition")

    def _perform_mstl_decomposition(self, data: pd.Series) -> Any:
        """
        Perform MSTL decomposition with optimized parameters.
        PURE MSTL mathematical logic without business adaptation.

        Args:
            data: Preprocessed time series

        Returns:
            MSTL decomposition result
        """
        # All parameters ALREADY adapted in configDecomposition.py
        periods = self.config["periods"]
        windows = self.config["windows"]
        iterate = self.config["iterate"]
        lmbda = self.config.get("lmbda")

        # Prepare STL kwargs from flat configuration
        stl_kwargs = self._prepare_stl_kwargs()

        try:
            # Create MSTL object with adapted parameters
            mstl = MSTL(
                data,
                periods=periods,
                windows=windows,
                lmbda=lmbda,
                iterate=iterate,
                stl_kwargs=stl_kwargs,
            )

            # Perform decomposition
            import time

            start_time = time.time()

            result = mstl.fit()

            execution_time = time.time() - start_time

            # Log performance only for long operations
            if execution_time > PERFORMANCE_LOG_THRESHOLD:
                logging.info(
                    f"{self} - MSTL executed in {execution_time:.3f}s "
                    f"for {len(periods)} periods: {periods}"
                )
            else:
                logging.debug(
                    f"{self} - MSTL executed in {execution_time:.3f}s "
                    f"for {len(periods)} periods"
                )

            return result

        except Exception as e:
            raise RuntimeError(f"MSTL decomposition failed: {str(e)}") from e

    def _prepare_stl_kwargs(self) -> Dict[str, Any]:
        """
        Prepare STL kwargs from flat configuration.
        Simple transformation without business logic.

        Returns:
            STL kwargs for MSTL
        """
        # All parameters ALREADY adapted in configDecomposition.py
        return {
            "seasonal_deg": self.config["seasonal_deg"],
            "trend_deg": self.config["trend_deg"],
            "low_pass_deg": self.config.get(
                "low_pass_deg", DEFAULT_LOW_PASS_DEGREE
            ),
            "robust": self.config["robust"],
            "inner_iter": self.config["inner_iter"],
            "outer_iter": self.config["outer_iter"],
            "trend": self.config["trend"],
        }

    def _extract_mstl_components(
        self, mstl_result: Any
    ) -> Tuple[pd.Series, pd.Series, pd.Series, Dict[str, Any]]:
        """
        Extract MSTL decomposition components.
        Simple mathematical processing of results.

        Args:
            mstl_result: MSTL decomposition result

        Returns:
            Tuple: (trend, seasonal_combined, residual, additional_data)
        """
        try:
            # Extract main components
            trend = mstl_result.trend
            seasonal_components = mstl_result.seasonal
            residual = mstl_result.resid

            # Validate components
            if any(comp is None for comp in [trend, residual]):
                raise ValueError("MSTL decomposition returned None components")

            # Handle multiple seasonal components
            if isinstance(seasonal_components, pd.DataFrame):
                # Sum all seasonal components for combined seasonal
                seasonal_combined = seasonal_components.sum(axis=1)

                # Details for each period for additional_data
                seasonal_details = {}
                periods = self.config["periods"]
                for i, period in enumerate(periods):
                    if i < seasonal_components.shape[1]:
                        seasonal_details[f"seasonal_period_{period}"] = (
                            seasonal_components.iloc[:, i]
                        )
            else:
                # For case of single seasonal component
                seasonal_combined = seasonal_components
                # Safe access to periods array with validation
                periods = self.config.get("periods", [])
                if len(periods) > SINGLE_COLUMN_INDEX:
                    seasonal_details = {
                        f"seasonal_period_{periods[SINGLE_COLUMN_INDEX]}": seasonal_components
                    }
                else:
                    # Fallback if no periods defined
                    seasonal_details = {"seasonal_period_unknown": seasonal_components}

            # Prepare additional_data
            additional_data = {
                "periods_used": self.config["periods"],
                "n_periods": len(self.config["periods"]),
                "windows_used": self.config["windows"],
                "iterate": self.config["iterate"],
                "seasonal_components": seasonal_details,
                "method": "mstl",
                "version": __version__,
                "config_applied": {
                    "lambda": self.config.get("lmbda"),
                    "robust": self.config["robust"],
                    "stl_params": self._prepare_stl_kwargs(),
                },
            }

            return trend, seasonal_combined, residual, additional_data

        except AttributeError as e:
            raise ValueError(f"Invalid MSTL result format: {str(e)}") from e