"""
Periodicity detection method based on spectral analysis.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal

from pipeline.helpers.utils import validate_required_locals
from pipeline.timeSeriesProcessing.periodicity.methods.basePeriodicityMethod import (
    BasePeriodicityMethod,
)

__version__ = "1.1.0"


class SpectralMethod(BasePeriodicityMethod):
    """
    Periodicity detection method based on spectral analysis (FFT).

    Uses mathematically correct fast Fourier transform
    for analyzing the frequency spectrum of time series with optimized
    scipy.signal algorithms.
    """

    # Default configuration for spectral method
    DEFAULT_CONFIG = {
        **BasePeriodicityMethod.DEFAULT_CONFIG,
        # "peak_height_ratio": 0.1,   # adapted in configPeriodicity.py
        # "window_type": "hann",      # adapted in configPeriodicity.py
        # "detrend": True,            # adapted in configPeriodicity.py
        # "n_peaks": 10,              # adapted in configPeriodicity.py
        # "min_peak_distance": 2,     # adapted in configPeriodicity.py
        # "nperseg": None,            # adapted in configPeriodicity.py
        # "noverlap": None,           # adapted in configPeriodicity.py
        # "nfft": None,               # adapted in configPeriodicity.py
        # "scaling": "density",       # adapted in configPeriodicity.py
        # "use_welch": True,          # adapted in configPeriodicity.py
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize spectral method.

        Args:
            config: Configuration with parameters (fully adapted)
        """
        # Merge configuration with defaults
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged_config)

        # Validate spectral analysis specific parameters
        validate_required_locals(
            [
                "peak_height_ratio",
                "window_type",
                "detrend",
                "n_peaks",
                "min_peak_distance",
            ],
            self.config,
        )

    def __str__(self) -> str:
        """Standard string representation for logging."""
        return (
            f"SpectralMethod(window={self.config['window_type']}, "
            f"use_welch={self.config['use_welch']}, period_range=[{self.min_period}, {self.max_period}])"
        )

    def process(
        self, data: pd.Series, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect periodicity using spectral analysis.

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

            # Prepare data for spectral analysis
            data_prepared = self.prepare_frequency_data(data, method="fft")
            if data_prepared["status"] == "error":
                return data_prepared

            clean_data = data_prepared["data"]

            # Compute power spectrum with mathematical correctness
            spectrum_result = self._compute_spectrum_core(clean_data)
            if spectrum_result["status"] == "error":
                return spectrum_result

            frequencies = spectrum_result["frequencies"]
            power = spectrum_result["power"]
            spectral_metadata = spectrum_result["metadata"]

            # Find peaks with parameter validation
            peaks_result = self._find_spectral_peaks_core(frequencies, power)
            if peaks_result["status"] == "error":
                return peaks_result

            # Convert frequencies to periods with intelligent ranking
            periods = self._frequencies_to_periods_optimized(
                peaks_result["peak_frequencies"], peaks_result["peak_powers"], data
            )

            # Prepare metadata
            execution_time = time.time() - start_time
            additional_data = {
                "spectrum_info": {
                    **spectral_metadata,
                    "n_peaks_found": len(peaks_result["peak_frequencies"]),
                    "frequency_range": (
                        [float(frequencies[0]), float(frequencies[-1])]
                        if len(frequencies) > 0
                        else [0, 0]
                    ),
                },
                "detection_method": "spectral",
                "window_type": self.config["window_type"],
                "detrend_applied": self.config["detrend"],
                "algorithm_version": __version__,
                "use_welch": self.config["use_welch"],
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
            return self.handle_error(e, "Spectral periodicity detection")


    def _compute_spectrum_core(self, data: np.ndarray) -> Dict[str, Any]:
        """Core spectrum computation logic."""
        data_length = len(data)

        # Use adapted parameters from configPeriodicity.py
        if self.config["use_welch"]:
            # Welch method for better PSD estimation with adapted parameters

            # Get adapted parameters
            nperseg = self.config["nperseg"]
            noverlap = self.config["noverlap"]

            # Correct Nyquist frequency handling
            welch_kwargs = {
                "nperseg": nperseg,
                "noverlap": noverlap,
                "window": self.config["window_type"],
                "scaling": self.config["scaling"],
                "return_onesided": True,
            }

            nfft = self.config.get("nfft")  # None by default - let scipy choose
            if nfft is not None:
                welch_kwargs["nfft"] = nfft

            if self.config["detrend"]:
                welch_kwargs["detrend"] = "linear"

            frequencies, power = signal.welch(data, **welch_kwargs)

            method_used = "welch"
            metadata = {
                "nperseg": nperseg,
                "noverlap": noverlap,
                "nfft": nfft,
                "window_applied": self.config["window_type"],
            }
        else:
            # Periodogram for simpler analysis
            periodogram_kwargs = {
                "window": self.config["window_type"],
                "scaling": self.config["scaling"],
                "return_onesided": True,
            }

            if self.config["detrend"]:
                periodogram_kwargs["detrend"] = "linear"

            frequencies, power = signal.periodogram(data, **periodogram_kwargs)

            method_used = "periodogram"
            metadata = {"window_applied": self.config["window_type"]}

        # Normalize power for stability
        if np.max(power) > 0:
            power = power / np.max(power)

        # Metadata for diagnostics
        spectral_metadata = {
            **metadata,
            "method": method_used,
            "frequency_resolution": (
                float(frequencies[1] - frequencies[0]) if len(frequencies) > 1 else 0
            ),
            "nyquist_frequency": float(frequencies[-1]) if len(frequencies) > 0 else 0,
            "n_frequencies": len(frequencies),
            "max_power_frequency": (
                float(frequencies[np.argmax(power)]) if len(power) > 0 else 0
            ),
            "total_power": float(np.sum(power)),
        }

        return {
            "status": "success",
            "frequencies": frequencies,
            "power": power,
            "metadata": spectral_metadata,
        }


    def _find_spectral_peaks_core(
        self, frequencies: np.ndarray, power: np.ndarray
    ) -> Dict[str, Any]:
        """Core spectral peak finding logic."""
        # Validate periodicity parameters
        validation = self.validate_periodicity_params(
            threshold=self.config["peak_height_ratio"]
        )
        if validation["status"] == "error":
            return validation

        # Convert frequencies to periods for filtering
        with np.errstate(divide="ignore", invalid="ignore"):
            periods = 1.0 / frequencies

        # Filter by valid period range
        max_period = self.max_period if self.max_period else len(periods) * 2
        valid_idx = (
            np.isfinite(periods)
            & (periods >= self.min_period)
            & (periods <= max_period)
        )

        if not np.any(valid_idx):
            return {
                "status": "success",
                "peak_frequencies": np.array([]),
                "peak_powers": np.array([]),
            }

        valid_freqs = frequencies[valid_idx]
        valid_power = power[valid_idx]

        # Adaptive peak finding
        min_height = np.max(valid_power) * self.config["peak_height_ratio"]

        # Dynamic min_distance calculation
        min_distance = max(
            self.config["min_peak_distance"],
            len(valid_power) // 100,  # Adaptive distance
        )

        # Robust peak finding
        peaks, properties = signal.find_peaks(
            valid_power,
            height=min_height,
            distance=min_distance,
            prominence=min_height * 0.3,  # Additional criterion for stability
        )

        # Intelligent ranking by power
        if len(peaks) > 0:
            # Sort by power (descending)
            sorted_idx = np.argsort(-valid_power[peaks])
            peaks = peaks[sorted_idx][: self.config["n_peaks"]]

            peak_frequencies = valid_freqs[peaks]
            peak_powers = valid_power[peaks]
        else:
            peak_frequencies = np.array([])
            peak_powers = np.array([])

        return {
            "status": "success",
            "peak_frequencies": peak_frequencies,
            "peak_powers": peak_powers,
        }

    def _frequencies_to_periods_optimized(
        self, frequencies: np.ndarray, powers: np.ndarray, original_data: pd.Series
    ) -> List[Tuple[int, float]]:
        """
        Convert frequencies to periods with optimized algorithm.

        Args:
            frequencies: Found peak frequencies
            powers: Powers of corresponding frequencies
            original_data: Original data for additional validation

        Returns:
            List of periods with confidence
        """
        if len(frequencies) == 0:
            return []

        periods_list = []

        # Vectorized processing for performance
        for freq, power in zip(frequencies, powers):
            # Precise frequency to period conversion
            period_float = 1.0 / freq
            period = int(round(period_float))

            # Penalty for imprecise rounding (important for spectral analysis)
            rounding_error = abs(period_float - period) / period_float
            rounding_penalty = 1.0 - min(
                rounding_error * 3, 0.7
            )  # More strict for FFT

            # Base confidence from power accounting for rounding
            method_confidence = float(power) * rounding_penalty

            # Apply intelligent period validation from base class
            final_confidence = self.calculate_period_confidence(
                original_data, period, method_confidence
            )

            # Filter by threshold
            if final_confidence >= self.confidence_threshold:
                periods_list.append((period, final_confidence))

        # Use intelligent ranking from base class
        return self.rank_periods(periods_list)