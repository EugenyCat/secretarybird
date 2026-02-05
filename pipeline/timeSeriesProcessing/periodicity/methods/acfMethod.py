"""
Periodicity detection method based on autocorrelation function.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from statsmodels.tsa.stattools import acf

from pipeline.helpers.utils import validate_required_locals
from pipeline.timeSeriesProcessing.periodicity.methods.basePeriodicityMethod import (
    BasePeriodicityMethod,
)

__version__ = "1.1.0"


class ACFMethod(BasePeriodicityMethod):
    """
    Periodicity detection method based on autocorrelation function (ACF).

    Detects periodicity by analyzing peaks in the autocorrelation
    function of a time series. Uses mathematically correct algorithm
    with optimized computations.
    """

    # Default configuration for ACF method
    DEFAULT_CONFIG = {
        **BasePeriodicityMethod.DEFAULT_CONFIG,
        # "peak_prominence": 0.1,       # adapted in configPeriodicity.py
        # "correlation_threshold": 0.1, # adapted in configPeriodicity.py
        # "max_lags_ratio": 0.5,        # adapted in configPeriodicity.py
        # "use_fft": True,              # adapted in configPeriodicity.py
        # "min_peak_distance": 2,       # adapted in configPeriodicity.py
        # "bias_correction": True,      # adapted in configPeriodicity.py
        # "nlags_limit": 1000,          # adapted in configPeriodicity.py
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ACF method.

        Args:
            config: Configuration with parameters (fully adapted)
        """
        # Merge configuration with defaults
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged_config)

        # Validate ACF-specific parameters
        validate_required_locals(
            [
                "peak_prominence",
                "correlation_threshold",
                "max_lags_ratio",
                "use_fft",
                "min_peak_distance",
            ],
            self.config,
        )

    def __str__(self) -> str:
        """Standard string representation for logging."""
        return (
            f"ACFMethod(correlation_threshold={self.config['correlation_threshold']}, "
            f"use_fft={self.config['use_fft']}, period_range=[{self.min_period}, {self.max_period}])"
        )

    def process(
        self, data: pd.Series, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect periodicity using ACF.

        Args:
            data: Time series for analysis
            context: Processing context

        Returns:
            Standardized result with detected periods
        """
        start_time = time.time()

        # Extract context parameters
        context_params = self.extract_context_parameters(context)

        # Log analysis start
        self.log_analysis_start(data, context_params)

        try:
            # Validate input data
            validation = self.validate_input(data)
            if validation["status"] == "error":
                return validation

            # Prepare data for ACF analysis
            data_prepared = self.prepare_frequency_data(data, method="acf")
            if data_prepared["status"] == "error":
                return data_prepared

            clean_data = data_prepared["data"]

            # Compute ACF with mathematical correctness
            acf_result = self._compute_acf_core(clean_data)
            if acf_result["status"] == "error":
                return acf_result

            acf_values = acf_result["acf_values"]
            n_lags = acf_result["n_lags"]

            # Find peaks with parameter validation
            peaks_result = self._find_peaks_core(acf_values)
            if peaks_result["status"] == "error":
                return peaks_result

            # Process peaks with intelligent ranking
            periods = self._process_peaks_optimized(
                peaks_result["peaks"],
                peaks_result["peak_values"],
                peaks_result.get("peak_prominences", np.array([])),
                acf_values,
                data,
            )

            # Prepare metadata
            execution_time = time.time() - start_time
            additional_data = {
                "acf_values": self._limit_acf_values(acf_values),
                "n_lags_computed": n_lags,
                "detection_method": "acf",
                "effective_threshold": self.config["correlation_threshold"],
                "bias_correction": self.config["bias_correction"],
                "algorithm_version": __version__,
            }

            # Create standard response
            result = self.prepare_result(
                periods=periods,
                additional_data=additional_data,
                execution_time=execution_time,
            )

            # Log completion
            self.log_analysis_complete(result)

            return result

        except Exception as e:
            return self.handle_error(e, "ACF periodicity detection")


    def _compute_acf_core(self, data: np.ndarray) -> Dict[str, Any]:
        """Core ACF computation logic."""
        # Determine optimal number of lags
        data_length = len(data)

        # Mathematically justified constraints
        max_lags_from_ratio = int(self.config["max_lags_ratio"] * data_length)
        max_lags_from_period = self.max_period if self.max_period else data_length // 3
        max_lags_absolute = min(self.config["nlags_limit"], data_length - 1)

        n_lags = min(max_lags_from_ratio, max_lags_from_period, max_lags_absolute)

        # Minimum number of lags for correct analysis
        min_lags_required = max(self.min_period * 2, 10)
        n_lags = max(n_lags, min_lags_required)

        # Compute ACF with bias correction
        try:
            acf_values = acf(
                data,
                nlags=n_lags,
                fft=self.config["use_fft"],
                missing="raise",
                # Bias correction improves quality for short series
                adjusted=self.config["bias_correction"],
            )
        except Exception as e:
            # Fallback without bias correction for problematic data
            if self.config["bias_correction"]:
                acf_values = acf(
                    data,
                    nlags=n_lags,
                    fft=self.config["use_fft"],
                    missing="raise",
                    adjusted=False,
                )
            else:
                raise e

        return {"status": "success", "acf_values": acf_values, "n_lags": n_lags}


    def _find_peaks_core(self, acf_values: np.ndarray) -> Dict[str, Any]:
        """Core peak finding logic."""
        # Validate periodicity parameters
        validation = self.validate_periodicity_params(
            threshold=self.config["correlation_threshold"]
        )
        if validation["status"] == "error":
            return validation

        # Dynamic min_distance calculation
        min_distance = max(
            self.min_period,
            self.config["min_peak_distance"],
            len(acf_values) // 200,  # Adaptive distance for long series
        )

        # Find peaks with robust parameters
        peaks, properties = signal.find_peaks(
            acf_values,
            height=self.config["correlation_threshold"],
            prominence=self.config["peak_prominence"],
            distance=min_distance,
        )

        # Filter by valid period range
        max_period = self.max_period if self.max_period else len(acf_values)
        valid_mask = (peaks >= self.min_period) & (peaks <= max_period)
        valid_peaks = peaks[valid_mask]

        # Extract values and prominences
        if len(valid_peaks) > 0:
            peak_values = acf_values[valid_peaks]

            # Safe extraction of prominences
            if "prominences" in properties and len(properties["prominences"]) > 0:
                peak_prominences = properties["prominences"][valid_mask]
            else:
                peak_prominences = (
                    np.ones_like(valid_peaks) * self.config["peak_prominence"]
                )
        else:
            peak_values = np.array([])
            peak_prominences = np.array([])

        return {
            "status": "success",
            "peaks": valid_peaks,
            "peak_values": peak_values,
            "peak_prominences": peak_prominences,
        }

    def _process_peaks_optimized(
        self,
        peaks: np.ndarray,
        peak_values: np.ndarray,
        peak_prominences: np.ndarray,
        acf_values: np.ndarray,
        original_data: pd.Series,
    ) -> List[Tuple[int, float]]:
        """
        Process found peaks with optimized algorithm.

        Args:
            peaks: Peak indices
            peak_values: ACF values at peaks
            peak_prominences: Peak prominences
            acf_values: Full ACF values
            original_data: Original data for additional validation

        Returns:
            List of periods with confidence
        """
        if len(peaks) == 0:
            return []

        periods = []

        # Vectorized processing for performance
        for i, (peak, value, prominence) in enumerate(
            zip(peaks, peak_values, peak_prominences)
        ):
            # Base confidence from ACF value
            base_confidence = float(value)

            # Prominence factor (normalized)
            prominence_factor = min(1.0, float(prominence) / 0.2)

            # Combined confidence
            method_confidence = base_confidence * prominence_factor

            # Apply intelligent period validation from base class
            final_confidence = self.calculate_period_confidence(
                original_data, int(peak), method_confidence
            )

            # Filter by threshold
            if final_confidence >= self.confidence_threshold:
                periods.append((int(peak), final_confidence))

        # Use intelligent ranking from base class
        return self.rank_periods(periods)

    def _limit_acf_values(self, acf_values: np.ndarray) -> List[float]:
        """
        Optimized limitation of ACF values for memory efficiency.

        Args:
            acf_values: Full ACF values array

        Returns:
            List of limited ACF values
        """
        max_points = 200

        if len(acf_values) <= max_points:
            return [float(val) for val in acf_values]

        # Optimized thinning strategy
        first_points = min(50, max_points // 2)
        remaining_points = max_points - first_points

        # First points (important for short periods)
        result: List[float] = [float(val) for val in acf_values[:first_points]]

        # Uniform thinning of the rest
        if len(acf_values) > first_points:
            indices = np.linspace(
                first_points, len(acf_values) - 1, remaining_points, dtype=int
            )
            result.extend([float(val) for val in acf_values[indices]])

        return result