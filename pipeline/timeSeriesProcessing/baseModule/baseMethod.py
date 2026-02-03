import logging
import time
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Dict, Optional, Protocol, Union

import pandas as pd

from pipeline.database.clickHouseConnection import ClickHouseConnection

"""
Common base classes for time series processing modules.

Contains base class BaseTimeSeriesMethod, which eliminates code duplication
between all time series analysis methods (Analyzer, Periodicity, Decomposition, etc.).
"""


class BaseTimeSeriesMethod(ABC):
    """
    Common base class for all time series processing methods.

    Eliminates code duplication between baseAnalysisMethod, basePeriodicityMethod
    and future base classes (Decomposition, OutlierRemover, Features).

    Contains only common logic:
    - Input data validation
    - Context parameter extraction
    - Clean data preparation
    - Standard metadata and response creation
    - Error handling and logging

    Specific functionality remains in child classes according to SOLID principles.
    """

    # 🔍 DEBUG TRACING: Class variable for temporary debug helper
    _trace_helper = None

    # Default base configurations (can be overridden in child classes)
    DEFAULT_CONFIG = {
        "return_detailed_metadata": False,  # Common interface setting
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base method.

        Args:
            config: Method configuration
        """
        # Merge configuration with defaults
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.name = self.__class__.__name__

    def __str__(self) -> str:
        """Standard string representation for logging."""
        return f"{self.name}(config_keys={list(self.config.keys())})"

    @abstractmethod
    def process(
        self, data: pd.Series, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute time series processing.

        Args:
            data: Time series for processing
            context: Additional context

        Returns:
            Dict with standard format:
            {
                'status': 'success/error',
                'result': {...},
                'metadata': {...}
            }
        """
        pass

    def validate_input(
        self, data: pd.Series, min_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Common input data validation.

        Args:
            data: Time series for validation
            min_length: Minimum data length

        Returns:
            Dict with validation result
        """
        try:
            if not isinstance(data, pd.Series):
                return self._create_error_response(
                    f"Expected pd.Series, got {type(data)}"
                )

            if len(data) == 0:
                return self._create_error_response("Empty series provided")

            if data.isnull().all():
                return self._create_error_response("All values in the series are null")

            # Check minimum length (if specified)
            if min_length is not None and len(data) < min_length:
                return self._create_error_response(
                    f"Series too short: {len(data)} < {min_length}"
                )

            # Check for missing values
            if data.isnull().sum() > 0:
                return self._create_error_response("Time series has missing values")

            return {"status": "success"}

        except Exception as e:
            return self._create_error_response(f"Validation error: {str(e)}")

    def extract_context_parameters(
        self, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract standard parameters from context.

        Args:
            context: Execution context

        Returns:
            Dict with extracted parameters
        """
        if not context:
            return {}

        # Common parameters that may be needed by all methods
        extracted = {
            "interval": context.get("interval", "unknown"),
            "data_source": context.get("data_source", "unknown"),
        }

        # Additional parameters stored separately
        additional_params = {
            k: v for k, v in context.items() if k not in ["interval", "data_source"]
        }

        if additional_params:
            extracted["additional_params"] = additional_params

        return extracted

    def prepare_clean_data(
        self, data: pd.Series, drop_na: bool = True, min_length: Optional[int] = None
    ) -> pd.Series:
        """
        Prepare clean data.

        Args:
            data: Original time series
            drop_na: Whether to remove NaN values
            min_length: Minimum length after cleaning

        Returns:
            Clean time series

        Raises:
            ValueError: If data fails validation
        """
        clean_data = data.dropna() if drop_na else data

        if min_length is not None and len(clean_data) < min_length:
            raise ValueError(
                f"Insufficient data after cleaning: {len(clean_data)} < {min_length}"
            )

        return clean_data

    def create_standard_metadata(
        self,
        data: pd.Series,
        context_params: Dict[str, Any],
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create standard metadata.

        Args:
            data: Time series
            context_params: Parameters from context
            additional_metadata: Additional metadata

        Returns:
            Dict with standard metadata
        """
        metadata = {
            "method": self.name,
            "data_length": len(data),
            "missing_values": int(data.isnull().sum()),
            "missing_ratio": data.isnull().sum() / len(data) if len(data) > 0 else 0.0,
            "interval": context_params.get("interval", "unknown"),
        }

        # Add detailed metadata if enabled
        if self.config.get("return_detailed_metadata", False):
            metadata["parameters_used"] = self.config.copy()
            metadata["data_source"] = context_params.get("data_source", "unknown")

        # Add additional metadata
        if additional_metadata:
            metadata.update(additional_metadata)

        return metadata

    def handle_error(
        self,
        error: Exception,
        operation: str,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Standardized error handling.

        Args:
            error: Exception
            operation: Operation name where error occurred
            additional_context: Additional error context

        Returns:
            Dict with standard error format
        """
        error_msg = f"Error in {operation}: {str(error)}"
        logging.error(f"{self} - {error_msg}", exc_info=True)

        error_response = {
            "status": "error",
            "message": error_msg,
            "metadata": {
                "method": self.name,
                "operation": operation,
                "error_type": type(error).__name__,
            },
        }

        if additional_context:
            error_response["metadata"].update(additional_context)

        return error_response

    def create_success_response(
        self,
        result: Dict[str, Any],
        data: pd.Series,
        context_params: Dict[str, Any],
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create standard success response.

        Args:
            result: Processing results
            data: Time series
            context_params: Context parameters
            additional_metadata: Additional metadata

        Returns:
            Dict with standard success response format
        """
        return {
            "status": "success",
            "result": result,
            "metadata": self.create_standard_metadata(
                data, context_params, additional_metadata
            ),
        }

    def _create_error_response(self, message: str) -> Dict[str, Any]:
        """Create standard error response."""
        return {"status": "error", "message": message}

    def log_analysis_start(
        self, data: pd.Series, context_params: Dict[str, Any]
    ) -> None:
        """Standard logging of analysis start."""
        logging.debug(
            f"{self} - Starting analysis: length={len(data)}, "
            f"interval={context_params.get('interval', 'unknown')}"
        )

    def log_analysis_complete(self, result: Dict[str, Any]) -> None:
        """Standard logging of analysis completion."""
        if isinstance(result, dict) and result.get("status") == "success":
            logging.debug(f"{self} - Analysis completed successfully")
        else:
            logging.warning(f"{self} - Analysis completed with issues")