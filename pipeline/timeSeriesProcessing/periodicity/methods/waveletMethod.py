"""
Periodicity detection method based on wavelet transform.

Uses continuous wavelet transform (CWT) for time-frequency characteristics analysis.
Optimized for working with PyWavelets and scipy.signal for maximum performance.

Version: 1.1.0
Author: Time Series Processing Team
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pywt
from scipy import signal

from pipeline.helpers.utils import validate_required_locals
from pipeline.timeSeriesProcessing.periodicity.methods.basePeriodicityMethod import (
    BasePeriodicityMethod,
)

__version__ = "1.1.0"


class WaveletMethod(BasePeriodicityMethod):
    """
    Periodicity detection method based on wavelet transform.

    Uses continuous wavelet transform (CWT) for analyzing time-frequency
    characteristics of time series with mathematically correct scale-to-frequency mapping.
    """

    # Wavelet-specific configuration with inheritance from base class
    DEFAULT_CONFIG = {
        **BasePeriodicityMethod.DEFAULT_CONFIG,
        # "wavelet": "cmor1.5-1.0",     # Complex Morlet wavelet;           Adapted in configPeriodicity
        # "n_scales": 50,               # Number of scales;                 Adapted in configPeriodicity
        # "scale_distribution": "log",  # log or linear;                    Adapted in configPeriodicity
        # "power_threshold": 0.1,       # Threshold for peak finding;       Adapted in configPeriodicity
        # "n_peaks": 5,                 # Maximum number of peaks;          Adapted in configPeriodicity
        # "sampling_period": 1.0,       # Sampling period;                  Adapted in configPeriodicity
        # "min_prominence": 0.3,        # Minimum peak height;              Adapted in configPeriodicity
        # "use_cone_of_influence": True,  # Use COI;                        Adapted in configPeriodicity
        # "scale_to_period_factor": 1.03,  # Conversion coefficient;        Adapted in configPeriodicity
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize wavelet method.

        Args:
            config: Configuration with adapted parameters
        """
        super().__init__(config)
        validate_required_locals(["wavelet", "n_scales"], self.config)

        self.logger = logging.getLogger(f"[Periodicity][{self.name}]")
        self._validate_wavelet_config()

    def __str__(self) -> str:
        """Standard string representation for logging."""
        return (
            f"{self.name}(wavelet={self.config['wavelet']}, "
            f"n_scales={self.config['n_scales']}, "
            f"period_range=[{self.min_period}, {self.max_period}])"
        )

    def _validate_wavelet_config(self) -> None:
        """Validate wavelet-specific configuration."""
        # Check wavelet availability
        available_wavelets = pywt.wavelist()
        if self.config["wavelet"] not in available_wavelets:
            # Fallback to proven wavelet
            self.config["wavelet"] = "cmor1.5-1.0"
            self.logger.warning(f"Unknown wavelet, using cmor1.5-1.0")

        # Validate scaling parameters
        if self.config["n_scales"] <= 0:
            raise ValueError("n_scales must be positive")
        if not 0 < self.config["power_threshold"] < 1:
            raise ValueError("power_threshold must be in range (0, 1)")

    def process(
        self, data: pd.Series, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect periodicity using wavelet analysis.

        Args:
            data: Time series for analysis
            context: Processing context

        Returns:
            Standardized result with detected periods
        """
        # Validate input data
        validation = self.validate_input(data)
        if validation["status"] == "error":
            return validation

        # Extract context parameters
        context_params = self.extract_context_parameters(context)

        # Log analysis start
        self.log_analysis_start(data, context_params)

        prepared_data = self.prepare_clean_data(data, drop_na=True)

        # Adapt parameters to data length
        self._adapt_scales_to_data_length(len(prepared_data))

        # Compute wavelet transform
        scales, power = self._compute_cwt_spectrum(prepared_data)

        # Find dominant scales
        dominant_scales, dominant_powers = self._find_dominant_scales(scales, power)

        # Convert to periods accounting for mathematical correctness
        periods = self._scales_to_periods(dominant_scales, dominant_powers)

        # Rank periods
        ranked_periods = self.rank_periods(
            periods, self.config["max_periods_returned"]
        )


        # Prepare result
        result = self.prepare_result(
            periods=ranked_periods,
            additional_data={
                "wavelet_info": {
                    "wavelet_type": self.config["wavelet"],
                    "n_scales_used": self.config["n_scales"],
                    "scale_distribution": self.config["scale_distribution"],
                    "power_threshold": self.config["power_threshold"],
                    "cone_of_influence": self.config["use_cone_of_influence"],
                },
                "detection_method": "wavelet",
            },
        )

        # Log completion
        self.log_analysis_complete(result)

        return result

    def _adapt_scales_to_data_length(self, data_length: int) -> None:
        """Adapt scaling parameters to data length."""
        # Limit maximum period
        if self.max_period is None:
            self.max_period = min(data_length // 2, 100)
        else:
            self.max_period = min(self.max_period, data_length // 2)

        # Adapt number of scales
        max_possible_scales = int(self.max_period - self.min_period + 1)
        self.config["n_scales"] = min(self.config["n_scales"], max_possible_scales)

    def _compute_cwt_spectrum(self, data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute CWT power spectrum with mathematically correct parameters.

        Args:
            data: Prepared data

        Returns:
            Tuple[scales, power_spectrum]
        """
        # Standardize data for wavelet analysis
        values = np.asarray(data.values)
        standardized = (values - np.mean(values)) / (np.std(values) + 1e-10)

        # Generate scales
        scales = self._generate_optimal_scales(len(data))

        # Compute CWT with error handling
        try:
            coefficients, _ = pywt.cwt(
                standardized,
                scales,
                self.config["wavelet"],
                sampling_period=self.config["sampling_period"],
            )

            # Power spectrum (Overflow protection: clip values to sqrt(float64_max) limit before squaring)
            power_spectrum = np.clip(np.abs(coefficients), 0, 1e154) ** 2

            # Time averaging to get global spectrum
            global_power = np.mean(power_spectrum, axis=1)

            # Normalization
            max_power = np.max(global_power)
            if max_power > 0:
                global_power = global_power / max_power

            return scales, global_power

        except Exception as e:
            self.logger.error(f"CWT error: {e}")
            raise ValueError(f"Wavelet transform error: {e}")

    def _generate_optimal_scales(self, data_length: int) -> np.ndarray:
        """
        Generate optimal scales accounting for mathematical constraints.

        Args:
            data_length: Data length

        Returns:
            Array of scales
        """
        # Determine scale range
        min_scale = max(2, self.min_period / 2)
        max_scale = min(self.max_period, data_length // 4)

        # Check range correctness
        if min_scale >= max_scale:
            min_scale = 2
            max_scale = min(data_length // 4, 50)

        # Generate scales
        if self.config["scale_distribution"] == "log":
            scales = np.logspace(
                np.log10(min_scale), np.log10(max_scale), self.config["n_scales"]
            )
        else:  # linear
            scales = np.linspace(min_scale, max_scale, self.config["n_scales"])

        return scales

    def _find_dominant_scales(
        self, scales: np.ndarray, power: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find dominant scales using scipy.signal.find_peaks.

        Args:
            scales: Array of scales
            power: Power spectrum

        Returns:
            Tuple[dominant_scales, dominant_powers]
        """
        if len(scales) == 0 or len(power) == 0:
            return np.array([]), np.array([])

        # Adaptive threshold for peak finding
        min_height = np.max(power) * self.config["power_threshold"]

        # Minimum distance between peaks
        min_distance = max(1, len(scales) // 20)

        # Find peaks with scipy.signal.find_peaks
        peaks, properties = signal.find_peaks(
            power,
            height=min_height,
            distance=min_distance,
            prominence=min_height * self.config["min_prominence"],
        )

        if len(peaks) == 0:
            return np.array([]), np.array([])

        # Sort by power and select top N
        sorted_indices = np.argsort(-power[peaks])
        selected_peaks = peaks[sorted_indices][: self.config["n_peaks"]]

        return scales[selected_peaks], power[selected_peaks]

    def _scales_to_periods(
        self, scales: np.ndarray, powers: np.ndarray
    ) -> List[Tuple[int, float]]:
        """
        Convert scales to periods with mathematically correct mapping.

        Args:
            scales: Dominant scales
            powers: Corresponding powers

        Returns:
            List of periods with confidence estimates
        """
        if len(scales) == 0:
            return []

        periods_list = []

        # Mathematically correct conversion coefficient
        scale_factor = self._get_scale_to_period_factor()

        for scale, power in zip(scales, powers):
            # Convert scale to period
            period_float = scale * scale_factor
            period = int(round(period_float))

            # Check range
            if period < self.min_period or period > self.max_period:
                continue

            # Base confidence from power
            base_confidence = float(power)

            # Penalty for imprecise rounding
            rounding_error = abs(period_float - period) / period_float
            rounding_penalty = max(0.5, 1.0 - rounding_error * 2)

            # Final confidence estimate
            final_confidence = self.calculate_period_confidence(
                pd.Series(np.zeros(period * 3)),  # Dummy data for calculation
                period,
                base_confidence * rounding_penalty,
            )

            if final_confidence >= self.confidence_threshold:
                periods_list.append((period, final_confidence))

        return periods_list

    def _get_scale_to_period_factor(self) -> float:
        """
        Get mathematically correct scale-to-period conversion coefficient.

        Returns:
            Conversion coefficient
        """
        # Coefficients for different wavelet types
        wavelet_factors = {
            "cmor": 1.03,  # Complex Morlet
            "morl": 1.03,  # Morlet
            "mexh": 1.0,  # Mexican hat
            "cgau": 1.0,  # Complex Gaussian
            "fbsp": 1.0,  # Frequency B-spline
            "shan": 1.0,  # Shannon
        }

        # Determine base wavelet type
        wavelet_base = self.config["wavelet"].split("-")[0].lower()

        # Return coefficient or default value
        return wavelet_factors.get(wavelet_base, self.config["scale_to_period_factor"])