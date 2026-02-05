"""
Base class for periodicity detection methods.
"""

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import signal

from pipeline.timeSeriesProcessing.baseModule.baseMethod import BaseTimeSeriesMethod
from pipeline.helpers.utils import validate_required_locals

__version__ = "1.1.0"


class BasePeriodicityMethod(BaseTimeSeriesMethod):
    """
    Base class for periodicity detection methods.

    Inherits common functionality from BaseTimeSeriesMethod.
    Defines specific interface and functionality for periodicity detection methods.
    """

    # Standard default configurations for periodicity
    DEFAULT_CONFIG = {
        **BaseTimeSeriesMethod.DEFAULT_CONFIG,
        # "min_period": 2,                  # Adapted in configPeriodicity
        # "max_period": None,               # Adapted in configPeriodicity
        # "confidence_threshold": 0.1,      # Adapted in configPeriodicity
        "min_data_length": 6,  # Mathematical minimum for periodicity ≥2
        "max_missing_ratio": 0.5,  # Reasonable threshold for data quality
        "max_periods_returned": 10,  # Reasonable limit for performance
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize periodicity detection method.

        Args:
            config: Method configuration (must be fully adapted)

        Raises:
            ValueError: If configuration is missing or incorrect
        """
        if not config:
            raise ValueError(
                f"Configuration is required for {self.__class__.__name__}. "
                f"Use build_config_from_properties() to generate configuration."
            )

        # Merge configuration with periodicity defaults
        merged_config = {**self.DEFAULT_CONFIG, **config}
        super().__init__(merged_config)

        # Validate base periodicity parameters
        validate_required_locals(["min_period", "confidence_threshold"], self.config)

        # Extract base parameters
        self.min_period = self.config["min_period"]
        self.max_period = self.config["max_period"]
        self.confidence_threshold = self.config["confidence_threshold"]

        # Validate period range
        if self.max_period is not None and self.max_period <= self.min_period:
            raise ValueError(
                f"max_period ({self.max_period}) must be greater than "
                f"min_period ({self.min_period})"
            )

    def __str__(self) -> str:
        """Standard string representation for periodicity logging."""
        return (
            f"{self.name}(min_period={self.min_period}, max_period={self.max_period})"
        )

    @abstractmethod
    def process(
        self, data: pd.Series, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect periodicity in time series.

        Args:
            data: Time series for analysis
            context: Context with additional information

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
        Extended input data validation accounting for periodicity specifics.

        Args:
            data: Time series for validation
            min_length: Minimum data length

        Returns:
            Dict with validation result
        """
        # Minimum length for periodicity detection
        min_len = min_length or max(self.min_period * 3, self.config["min_data_length"])

        # Use base validation
        validation_result = super().validate_input(data, min_len)
        if validation_result["status"] == "error":
            return validation_result

        # Additional checks for periodicity
        try:
            # Check for constancy (critical for periodicity)
            if data.nunique() == 1:
                return self._create_error_response(
                    "Data is constant, no periodicity possible"
                )

            # Check for sufficient variability
            if data.nunique() < 3:
                return self._create_error_response(
                    f"Insufficient variability: only {data.nunique()} unique values"
                )

            return {"status": "success"}

        except Exception as e:
            return self._create_error_response(
                f"Periodicity validation error: {str(e)}"
            )

    def extract_context_parameters(
        self, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract context parameters accounting for periodicity specifics.

        Args:
            context: Execution context

        Returns:
            Dict with extracted parameters
        """
        # Get base parameters
        base_params = super().extract_context_parameters(context)

        # Add periodicity-specific parameters
        if context:
            base_params["sampling_rate"] = context.get("sampling_rate", 1.0)

        return base_params

    def validate_periodicity_params(self, **params) -> Dict[str, Any]:
        """
        Validate periodicity-specific parameters.

        Args:
            **params: Parameters for validation

        Returns:
            Dict with validation result
        """
        try:
            # Validate periods
            if "periods" in params:
                periods = params["periods"]
                if not isinstance(periods, (list, np.ndarray)):
                    return self._create_error_response("periods must be list or array")

                if len(periods) == 0:
                    return self._create_error_response("periods cannot be empty")

                # Check period range
                for period in periods:
                    if period < self.min_period:
                        return self._create_error_response(
                            f"Period {period} < min_period {self.min_period}"
                        )
                    if self.max_period and period > self.max_period:
                        return self._create_error_response(
                            f"Period {period} > max_period {self.max_period}"
                        )

            # Validate thresholds
            if "threshold" in params:
                threshold = params["threshold"]
                if not isinstance(threshold, (int, float)):
                    return self._create_error_response("threshold must be numeric")
                if threshold < 0 or threshold > 1:
                    return self._create_error_response("threshold must be in [0, 1]")

            return {"status": "success"}

        except Exception as e:
            return self._create_error_response(f"Parameter validation error: {str(e)}")

    def prepare_frequency_data(
        self, data: pd.Series, method: str = "fft"
    ) -> Dict[str, Any]:
        """
        Prepare data for frequency analysis.

        Args:
            data: Time series
            method: Preparation method ("fft", "cwt", "acf")

        Returns:
            Dict with prepared data
        """
        try:
            clean_data = self.prepare_clean_data(data)

            if method == "fft":
                # Detrend for FFT
                values = signal.detrend(np.array(clean_data.values))
                return {
                    "status": "success",
                    "data": values,
                    "length": len(values),
                    "preprocessing": "detrended",
                }

            elif method == "cwt":
                # Standardization for wavelet analysis
                values = np.array(clean_data.values)
                mean = np.mean(values)
                std = np.std(values)

                if std > 0:
                    standardized = (values - mean) / std
                else:
                    standardized = values - mean

                return {
                    "status": "success",
                    "data": standardized,
                    "length": len(standardized),
                    "preprocessing": "standardized",
                }

            elif method == "acf":
                # Minimal processing for ACF
                return {
                    "status": "success",
                    "data": np.array(clean_data.values),
                    "length": len(clean_data),
                    "preprocessing": "none",
                }

            else:
                return self._create_error_response(f"Unknown method: {method}")

        except Exception as e:
            return self.handle_error(e, "frequency data preparation")

    def calculate_period_confidence(
        self, data: pd.Series, period: int, method_score: float
    ) -> float:
        """
        Calculate confidence in detected period.

        Args:
            data: Time series
            period: Detected period
            method_score: Score from specific method

        Returns:
            Normalized confidence score [0, 1]
        """
        if period <= 0 or period >= len(data):
            return 0.0

        # Base confidence from method
        confidence = float(method_score)

        # Penalties for extreme periods
        if period < 3:
            confidence *= 0.7
        elif period == 2:
            confidence *= 0.5

        # Periods close to series length
        length_ratio = period / len(data)
        if length_ratio > 0.5:
            penalty = 1.0 - (length_ratio - 0.5)
            confidence *= penalty

        # Check for period multiplicity in data
        n_cycles = len(data) / period
        if n_cycles < 2.0:
            confidence *= 0.8

        return max(0.0, min(1.0, confidence))

    def rank_periods(
        self, periods: List[Tuple[int, float]], max_periods: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Rank periods by confidence.

        Args:
            periods: List of (period, confidence)
            max_periods: Maximum number of periods

        Returns:
            Sorted list of best periods
        """
        if not periods:
            return []

        # Sort by confidence (descending)
        sorted_periods = sorted(periods, key=lambda x: x[1], reverse=True)

        # Filter by confidence threshold
        filtered_periods = [
            (period, conf)
            for period, conf in sorted_periods
            if conf >= self.confidence_threshold
        ]

        # Limit count
        max_count = min(max_periods, self.config["max_periods_returned"])
        return filtered_periods[:max_count]

    def create_standard_metadata(
        self,
        data: pd.Series,
        context_params: Dict[str, Any],
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create standard metadata for periodicity result.

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

        # Add periodicity-specific metadata
        metadata.update(
            {
                "period_range": [self.min_period, self.max_period],
                "confidence_threshold": self.confidence_threshold,
            }
        )

        # Add detailed metadata if enabled
        if self.config.get("return_detailed_metadata", False):
            metadata["sampling_rate"] = context_params.get("sampling_rate", 1.0)

        return metadata

    def prepare_result(
        self,
        periods: List[Tuple[int, float]],
        additional_data: Optional[Dict[str, Any]] = None,
        execution_time: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Prepare standardized result.

        Args:
            periods: List of detected periods
            additional_data: Additional data
            execution_time: Execution time

        Returns:
            Standardized result
        """
        # Rank periods
        ranked_periods = self.rank_periods(periods)

        result = {
            "periods": ranked_periods,
            "n_periods_found": len(ranked_periods),
            "execution_time": execution_time,
        }

        if additional_data:
            result.update(additional_data)

        return {
            "status": "success",
            "result": result,
            "metadata": {
                "method": self.name,
                "confidence_threshold": self.confidence_threshold,
                "period_range": [self.min_period, self.max_period],
            },
        }

    def log_analysis_start(
        self, data: pd.Series, context_params: Dict[str, Any]
    ) -> None:
        """
        Log start of periodicity analysis.

        Args:
            data: Time series
            context_params: Context parameters
        """
        logging.info(
            f"Starting periodicity analysis: length={len(data)}, "
            f"interval={context_params.get('interval', 'unknown')}, "
            f"range=[{self.min_period}, {self.max_period}]"
        )

    def log_analysis_complete(self, result: Dict[str, Any]) -> None:
        """
        Log completion of periodicity analysis.

        Args:
            result: Analysis result
        """
        if result["status"] == "success":
            n_periods = result["result"]["n_periods_found"]
            exec_time = result["result"].get("execution_time", 0.0)

            logging.info(
                f"Analysis complete: {n_periods} periods found, "
                f"time={exec_time:.3f}s"
            )
        else:
            logging.error(f"Analysis failed: {result['message']}")

    def get_used_parameters(self) -> Dict[str, Any]:
        """
        Get method's used parameters.

        Returns:
            Dict with method parameters
        """
        return {
            "name": self.name,
            "min_period": self.min_period,
            "max_period": self.max_period,
            "confidence_threshold": self.confidence_threshold,
            "version": __version__,
        }