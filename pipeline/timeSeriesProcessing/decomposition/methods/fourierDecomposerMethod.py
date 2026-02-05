"""
Fourier method for time series decomposition.

Implements harmonic analysis via FFT to extract periodic components.
Optimal for stationary data with clear periodicity and low noise.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq

from pipeline.helpers.utils import validate_required_locals
from pipeline.timeSeriesProcessing.decomposition.methods.baseDecomposerMethod import (
    BaseDecomposerMethod,
)

__version__ = "1.1.0"


class FourierDecomposerMethod(BaseDecomposerMethod):
    """
    Fourier method for time series decomposition.

    Uses FFT and harmonic analysis to extract periodic
    components. Especially effective for stationary data with clear
    periodicity (quality > 0.7) and low noise (< 0.05).
    """

    # Specific configurations for Fourier method
    DEFAULT_CONFIG = {
        **BaseDecomposerMethod.DEFAULT_CONFIG,
        # "period": None,        # [AUTO] from configDecomposition.py (adapted in configDecomposition)
        # "max_harmonics": 10,   # Maximum harmonics for analysis (adapted in configDecomposition)
        # "use_aic": True,       # Use AIC for optimization (adapted in configDecomposition)
        # "detrend_first": True,     # Detrend before analysis (adapted in configDecomposition)
        # "frequency_threshold": 0.1,    # Frequency significance threshold (adapted in configDecomposition)
        # "prominence_factor": 0.2,      # Multiplier for prominence detection (adapted in configDecomposition)
        # "peak_threshold": 0.05,        # Minimum peak height (adapted in configDecomposition)
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Fourier decomposition method.

        Args:
            config: Method configuration (must be fully adapted)
        """
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged_config)

        # Validate Fourier-specific parameters
        validate_required_locals(
            [
                "period",
                "max_harmonics",
                "use_aic",
                "detrend_first",
                "frequency_threshold",
                "prominence_factor",
                "peak_threshold",
            ],
            self.config,
        )

    def __str__(self) -> str:
        """Standard string representation for logging."""
        return (
            f"FourierDecomposerMethod(harmonics={self.config['max_harmonics']}, "
            f"aic={self.config['use_aic']}, "
            f"detrend={self.config['detrend_first']})"
        )

    def _validate_config(self) -> None:
        """
        Validate Fourier method configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate max_harmonics
        if not (1 <= self.config["max_harmonics"] <= 100):
            raise ValueError(
                f"max_harmonics must be in [1, 100], got: {self.config['max_harmonics']}"
            )

        # Validate frequency_threshold
        if not (0.0 <= self.config["frequency_threshold"] <= 1.0):
            raise ValueError(
                f"frequency_threshold must be in [0, 1], got: {self.config['frequency_threshold']}"
            )

    def process(
        self, data: pd.Series, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform Fourier decomposition of time series.

        Args:
            data: Time series for decomposition
            context: Context with additional information

        Returns:
            Dict with standard result format
        """
        try:
            # 1. CRITICAL fail-fast validation
            critical_validation = self.validate_input_critical(data)
            if critical_validation is not None:
                return critical_validation

            # 2. Standard input validation
            validation = self.validate_input(data)
            if validation["status"] == "error":
                return validation

            # Extract context parameters
            context_params = self.extract_context_parameters(context)

            # Preprocess data
            processed_data = self.preprocess_data(data)

            # Period from configuration (already adapted)
            period = self.config["period"]
            if not period or period < 2:
                return self.handle_error(
                    ValueError(f"Insufficient period for Fourier: {period}"),
                    "period validation",
                )

            # Detrend if required
            if self.config["detrend_first"]:
                detrended, trend_component = self._detrend_data(processed_data)
            else:
                detrended = processed_data
                trend_component = pd.Series(0, index=processed_data.index)

            # Perform Fourier analysis
            fourier_result = self._perform_fourier_analysis(detrended, period)

            seasonal_component = fourier_result["seasonal"]
            residual = fourier_result["residual"]
            harmonics_info = fourier_result["harmonics_info"]

            # Additional data for result
            additional_data = {
                "period_used": period,
                "harmonics_info": harmonics_info,
            }

            # Create standard response
            return self.prepare_decomposition_result(
                trend_component,
                seasonal_component,
                residual,
                data,
                context_params,
                additional_data,
            )

        except Exception as e:
            return self.handle_error(e, "Fourier decomposition")

    def _detrend_data(self, data: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Remove trend from time series.

        Args:
            data: Time series

        Returns:
            Tuple[detrended series, trend]
        """
        try:
            # Use scipy.signal.detrend for more correct detrending
            data_array = np.asarray(data.values, dtype=np.float64)
            detrended_values = signal.detrend(data_array, type="linear")
            detrended = pd.Series(detrended_values, index=data.index)

            # Restore trend
            trend = data - detrended

            # Safe slope calculation
            x_vals = np.arange(len(data), dtype=np.float64)
            trend_vals = np.asarray(trend.values, dtype=np.float64)
            slope = np.polyfit(x_vals, trend_vals, 1)[0]

            logging.info(
                f"{self} - Linear detrending: removed trend with slope={slope:.4f}"
            )

            return detrended, trend

        except Exception as e:
            logging.warning(
                f"{self} - Detrending error: {e}, using original data"
            )
            return data, pd.Series(0, index=data.index)

    def _perform_fourier_analysis(self, data: pd.Series, period: int) -> Dict[str, Any]:
        """
        Perform comprehensive Fourier analysis with mathematical correctness.

        Args:
            data: Detrended time series
            period: Main period

        Returns:
            Dict with seasonal component, residual and harmonics_info
        """
        n = len(data)

        # Memory-efficient FFT with correct typing
        data_array = np.asarray(data.values, dtype=np.float64)

        # FFT execution with correct typing
        fft_result = fft(data_array)
        fft_values = np.asarray(fft_result, dtype=np.complex128)
        frequencies = np.asarray(fftfreq(n, d=1.0), dtype=np.float64)

        # Calculate power spectral density
        power_spectrum = np.abs(fft_values) ** 2

        # Find dominant frequencies with prominence-based peak detection
        freq_mask = frequencies > 0
        positive_freqs = frequencies[freq_mask]
        positive_power = power_spectrum[freq_mask]

        # Find peaks with adaptive prominence (optimized)
        max_power = np.max(positive_power)
        prominence_threshold = max_power * self.config["prominence_factor"]
        height_threshold = max_power * self.config["peak_threshold"]

        peaks, properties = signal.find_peaks(
            positive_power,
            prominence=prominence_threshold,
            height=height_threshold,
        )

        # Limit number of harmonics
        max_harmonics = min(
            self.config["max_harmonics"],
            period // 2,  # Nyquist limit
            n // 4,  # Practical limit
            len(peaks),  # Available peaks
        )

        if max_harmonics == 0:
            logging.warning(f"{self} - No significant peaks found in spectrum")
            return {
                "seasonal": pd.Series(0, index=data.index),
                "residual": data,
                "harmonics_info": self._create_empty_harmonics_info(),
            }

        # Select top peaks by power
        top_peak_indices = np.argsort(positive_power[peaks])[-max_harmonics:]
        selected_peaks = peaks[top_peak_indices]
        selected_frequencies = positive_freqs[selected_peaks]
        selected_amplitudes = positive_power[selected_peaks]

        # Build seasonal component from selected harmonics
        seasonal_component = self._reconstruct_seasonal_from_harmonics(
            data, selected_frequencies, fft_values, frequencies
        )

        # Calculate residuals
        residual = data - seasonal_component

        # Calculate spectral entropy for quality assessment
        spectral_entropy = self._calculate_spectral_entropy(positive_power)

        # Safe phase extraction with correct typing
        positive_indices = np.where(freq_mask)[0]
        positive_fft = fft_values[positive_indices]
        selected_phases = np.angle(positive_fft[selected_peaks]).tolist()

        # Create harmonics_info
        harmonics_info = {
            "n_harmonics": len(selected_frequencies),
            "amplitudes": np.sqrt(selected_amplitudes).tolist(),
            "phases": selected_phases,
            "frequency_amplitudes": selected_amplitudes.tolist(),
            "dominant_frequencies": selected_frequencies.tolist(),
            "spectral_entropy": spectral_entropy,
            "power_spectrum_peak_ratio": (
                np.max(selected_amplitudes) / np.sum(positive_power)
                if np.sum(positive_power) > 0
                else 0.0
            ),
        }

        logging.info(
            f"{self} - Fourier analysis: {len(selected_frequencies)} harmonics, entropy={spectral_entropy:.3f}"
        )

        return {
            "seasonal": seasonal_component,
            "residual": residual,
            "harmonics_info": harmonics_info,
        }

    def _reconstruct_seasonal_from_harmonics(
        self,
        data: pd.Series,
        frequencies: np.ndarray,
        fft_values: np.ndarray,
        all_frequencies: np.ndarray,
    ) -> pd.Series:
        """
        Reconstruct seasonal component from selected harmonics.
        Vectorized version for improved performance.

        Args:
            data: Original series
            frequencies: Selected frequencies
            fft_values: FFT values
            all_frequencies: All frequencies

        Returns:
            Reconstructed seasonal component
        """
        n = len(data)
        t = np.arange(n)

        if len(frequencies) == 0:
            return pd.Series(np.zeros(n), index=data.index)

        # Vectorized frequency index search
        # Use broadcasting to calculate all distances at once
        distances = np.abs(all_frequencies[:, np.newaxis] - frequencies[np.newaxis, :])
        freq_indices = np.argmin(distances, axis=0)

        # Vectorized extraction of amplitudes and phases
        selected_fft_values = fft_values[freq_indices]
        amplitudes = 2.0 * np.abs(selected_fft_values) / n
        phases = np.angle(selected_fft_values)

        # Vectorized calculation of all harmonics
        # Create matrix (n_samples, n_frequencies) for broadcasting
        t_matrix = t[:, np.newaxis]  # (n, 1)
        freq_matrix = frequencies[np.newaxis, :]  # (1, n_freqs)
        phase_matrix = phases[np.newaxis, :]  # (1, n_freqs)
        amp_matrix = amplitudes[np.newaxis, :]  # (1, n_freqs)

        # Calculate all harmonics at once via broadcasting
        harmonic_matrix = amp_matrix * np.cos(
            2 * np.pi * freq_matrix * t_matrix + phase_matrix
        )

        # Sum over all frequencies
        seasonal_values = np.sum(harmonic_matrix, axis=1)

        return pd.Series(seasonal_values, index=data.index)

    def _calculate_spectral_entropy(self, power_spectrum: np.ndarray) -> float:
        """
        Calculate spectral entropy for quality assessment.

        Args:
            power_spectrum: Power spectral density

        Returns:
            Spectral entropy
        """
        # Normalize spectrum
        total_power = np.sum(power_spectrum)
        if total_power == 0:
            return 0.0

        normalized_spectrum = power_spectrum / total_power

        # Avoid log(0) using Boolean indexing
        valid_mask = normalized_spectrum > 0
        if not np.any(valid_mask):
            return 0.0

        normalized_spectrum = normalized_spectrum[valid_mask]

        # Calculate entropy
        entropy = -np.sum(normalized_spectrum * np.log2(normalized_spectrum))

        # Normalize to [0, 1]
        max_entropy = np.log2(len(normalized_spectrum))
        if max_entropy > 0:
            entropy = entropy / max_entropy

        return float(entropy)

    def _create_empty_harmonics_info(self) -> Dict[str, Any]:
        """Create empty harmonics_info structure."""
        return {
            "n_harmonics": 0,
            "amplitudes": [],
            "phases": [],
            "frequency_amplitudes": [],
            "dominant_frequencies": [],
            "spectral_entropy": 0.0,
            "power_spectrum_peak_ratio": 0.0,
        }