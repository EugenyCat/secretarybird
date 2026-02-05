"""
Configurations for time series periodicity detector.

=== IMPLEMENTS BaseConfigAdapter ===
Uses Template Method pattern to eliminate ~60% code duplication.

Critical features:
- max_period MUST be set BEFORE constraints (in _pre_constraint_setup)
- Spectral parameters are calculated AFTER max_period is set
- Weights are normalized at the end after all adaptations
"""

import logging
from copy import deepcopy
import math
import numpy as np
from scipy.stats import norm
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from pipeline.helpers.configs import InstrumentTypeConfig
from pipeline.timeSeriesProcessing.baseModule.baseConfigAdapter import BaseConfigAdapter
from pipeline.timeSeriesProcessing.preprocessingConfig import (
    DataLengthCategory,
    FrequencyCategory,
)

__version__ = "2.0.0"

# ===== CONSTANTS (NO CHANGES) =====

CRYPTO_PERIODICITY_VOLATILITY_THRESHOLDS = {
    "stable": 0.1,
    "variable": 0.3,
    "chaotic": 0.5,
}

CRYPTO_HIGH_VOLATILITY_ADJUSTMENTS = {
    "base.min_data_length": 1.5,
    "base.max_missing_ratio": 0.7,
    "acf.confidence_threshold": 0.7,
    "acf.peak_prominence": 0.8,
    "acf.correlation_threshold": 0.7,
    "acf.nlags_limit": 1.3,
    "acf.bias_correction": True,
    "spectral.confidence_threshold": 0.7,
    "spectral.peak_height_ratio": 0.8,
    "spectral.n_peaks": 1.5,
    "spectral.use_welch": True,
    "spectral.scaling": "density",
    "wavelet.confidence_threshold": 0.7,
    "wavelet.power_threshold": 0.8,
    "wavelet.n_peaks": 1.5,
    "wavelet.min_prominence": 0.6,
    "wavelet.use_cone_of_influence": True,
}

BASE = {
    "base": {
        # Box & Jenkins (1976): minimum 30 observations for ACF statistical significance
        # Ensures 2-3 full cycles for periods up to 10-15
        # Spectral analysis: minimum 32 points (2^5) for reasonable FFT quality
        # Wavelet analysis: minimum 50 points for cone of influence coverage (Torrence & Compo, 1998)
        "min_data_length": 30,
        # Stricter missing data threshold for reliable periodicity detection
        "max_missing_ratio": 0.2,
    },
    "acf": {
        "min_period": 2,
        "max_period": None,
        "confidence_threshold": 0.3,
        "peak_prominence": 0.1,
        "correlation_threshold": 0.3,
        "max_lags_ratio": 0.5,
        "use_fft": True,
        "min_peak_distance": 1,
        "bias_correction": True,
        "nlags_limit": 1000,
    },
    "spectral": {
        "min_period": 2,
        "max_period": None,
        "confidence_threshold": 0.3,
        "peak_height_ratio": 0.2,
        "window_type": "hann",
        "detrend": True,
        "n_peaks": 5,
        "min_peak_distance": 1,
        "nperseg": None,
        "noverlap": None,
        "nfft": None,
        "scaling": "density",
        "use_welch": True,
    },
    "wavelet": {
        "min_period": 2,
        "max_period": None,
        "confidence_threshold": 0.3,
        "wavelet": "cmor1.5-1.0",
        "n_scales": 100,
        "scale_distribution": "log",
        "power_threshold": 0.2,
        "n_peaks": 5,
        "sampling_period": 1.0,
        "min_prominence": 0.3,
        "use_cone_of_influence": True,
        # Theoretical scale-to-period conversion factor for cmor1.5-1.0 wavelet
        # Formula: period = scale * Fourier_factor * sampling_period
        # For different wavelet types (Torrence & Compo, 1998):
        #   cmor1.5-1.0: ~1.0 (central frequency = 1.5)
        #   morl: ~1.0
        #   mexh: ~1.58 (sqrt(5/2))
        #   cgau5: ~2.24 (sqrt(5))
        # Reference: Torrence & Compo (1998). "A Practical Guide to Wavelet Analysis"
        "scale_to_period_factor": 1.0,
    },
}

ACTIVE = {
    InstrumentTypeConfig.CRYPTO: ["base", "acf", "spectral", "wavelet"],
}

DEFAULT_STRUCTURE = {
    "methods": ["acf", "spectral", "wavelet"],
    "verify_with_stl": True,
    "weights": {"acf": 1.0, "spectral": 1.0, "wavelet": 1.0},
}

STRUCTURES = {
    "frequency": {
        FrequencyCategory.HIGH: {
            "methods": ["acf", "spectral", "wavelet"],
            "weights": {"acf": 0.5, "spectral": 1.0, "wavelet": 0.8},
        },
        FrequencyCategory.MEDIUM: {
            "methods": ["acf", "spectral", "wavelet"],
            "weights": {"acf": 0.7, "spectral": 1.0, "wavelet": 0.8},
        },
        FrequencyCategory.LOW: {
            "methods": ["acf", "spectral"],
            "weights": {"acf": 1.0, "spectral": 0.7},
        },
    },
    "length": {
        DataLengthCategory.TINY: {
            "methods": ["acf"],
            "verify_with_stl": False,
            "weights": {"acf": 1.0},
        },
        DataLengthCategory.SHORT: {
            "methods": ["acf", "spectral"],
            "verify_with_stl": False,
            "weights": {"acf": 1.0, "spectral": 0.7},
        },
        DataLengthCategory.SMALL: {"verify_with_stl": False},
    },
}

RULES = {
    "frequency": {
        FrequencyCategory.HIGH: [
            ("base.min_data_length", "*", 2.0),
            ("base.max_missing_ratio", "*", 0.7),
            ("acf.confidence_threshold", "=", 0.1),
            ("acf.peak_prominence", "=", 0.03),
            ("acf.correlation_threshold", "=", 0.1),
            ("acf.max_lags_ratio", "=", 0.3),
            # High frequency: reduce lags (short-term correlations dominant)
            # For 1-min/5-min data, significant correlations are at short lags
            # Multiplier 0.6 reduces computational waste on long lags
            ("acf.nlags_limit", "*", 0.6),
            ("spectral.confidence_threshold", "=", 0.1),
            ("spectral.peak_height_ratio", "=", 0.03),
            ("spectral.window_type", "=", "blackman"),
            ("spectral.n_peaks", "=", 15),
            ("spectral.scaling", "=", "spectrum"),
            # High frequency: increase resolution to distinguish close periods
            # Better frequency resolution needed to separate periods like 60min vs 65min
            # Multiplier 1.3 improves spectral resolution for high-frequency trading
            ("spectral.nperseg", "*", 1.3),
            # Note: spectral.noverlap rule removed - standard 50% overlap optimal (Welch, 1967)
            ("wavelet.confidence_threshold", "=", 0.1),
            ("wavelet.n_scales", "=", 150),
            ("wavelet.power_threshold", "=", 0.1),
            ("wavelet.n_peaks", "=", 10),
            ("wavelet.min_prominence", "*", 0.7),
        ],
        FrequencyCategory.MEDIUM: [
            ("base.min_data_length", "*", 1.5),
            ("base.max_missing_ratio", "*", 0.8),
            ("acf.confidence_threshold", "=", 0.12),
            ("acf.peak_prominence", "=", 0.05),
            ("acf.correlation_threshold", "=", 0.15),
            ("acf.max_lags_ratio", "=", 0.4),
            ("acf.nlags_limit", "*", 1.5),
            ("spectral.confidence_threshold", "=", 0.12),
            ("spectral.peak_height_ratio", "=", 0.1),
            ("spectral.n_peaks", "=", 8),
            ("spectral.scaling", "=", "density"),
            ("spectral.nperseg", "*", 0.9),
            # Note: spectral.noverlap rule removed - standard 50% overlap optimal (Welch, 1967)
            ("wavelet.confidence_threshold", "=", 0.12),
            ("wavelet.power_threshold", "=", 0.15),
            ("wavelet.n_peaks", "=", 8),
            ("wavelet.min_prominence", "*", 0.9),
        ],
        FrequencyCategory.LOW: [
            ("base.min_data_length", "*", 1.0),
            ("base.max_missing_ratio", "*", 1.2),
            ("acf.confidence_threshold", "=", 0.15),
            ("acf.peak_prominence", "=", 0.08),
            ("acf.correlation_threshold", "=", 0.2),
            ("acf.max_lags_ratio", "=", 0.6),
            ("acf.nlags_limit", "*", 0.8),
            ("acf.bias_correction", "=", True),
            ("spectral.confidence_threshold", "=", 0.15),
            ("spectral.peak_height_ratio", "=", 0.08),
            ("spectral.window_type", "=", "blackman"),
            ("spectral.n_peaks", "=", 5),
            ("spectral.min_peak_distance", "=", 2),
            ("spectral.use_welch", "=", True),
            ("spectral.nperseg", "*", 1.3),
            # Note: spectral.noverlap rule removed - standard 50% overlap optimal (Welch, 1967)
            ("wavelet.min_prominence", "*", 1.2),
        ],
    },
    "length": {
        DataLengthCategory.TINY: [
            # Box & Jenkins (1976): minimum 3 full cycles required for statistical significance
            # min_period=2 → 3 cycles × 2 = 6, but use 12 for robustness (4 cycles × 3)
            ("base.min_data_length", "=", 12),  # 3-4 cycles minimum
            ("base.max_missing_ratio", "*", 1.5),
            ("acf.confidence_threshold", "=", 0.05),
            ("acf.peak_prominence", "=", 0.02),
            ("acf.correlation_threshold", "=", 0.05),
            ("acf.max_lags_ratio", "=", 0.8),
            ("acf.nlags_limit", "=", 50),
            ("acf.bias_correction", "=", False),
        ],
        DataLengthCategory.SHORT: [
            ("base.min_data_length", "=", 8),
            ("base.max_missing_ratio", "*", 1.3),
            ("acf.confidence_threshold", "=", 0.05),
            ("acf.peak_prominence", "=", 0.02),
            ("acf.correlation_threshold", "=", 0.05),
            ("acf.max_lags_ratio", "=", 0.8),
            ("acf.nlags_limit", "=", 100),
            ("spectral.confidence_threshold", "=", 0.05),
            ("spectral.peak_height_ratio", "=", 0.02),
            ("spectral.n_peaks", "=", 10),
        ],
        DataLengthCategory.SMALL: [
            ("base.min_data_length", "=", 10),
            ("wavelet.n_scales", "=", 30),
            ("wavelet.use_cone_of_influence", "=", True),
            ("acf.nlags_limit", "=", 200),
        ],
        DataLengthCategory.LARGE: [
            ("acf.max_lags_ratio", "=", 0.1),
            ("acf.nlags_limit", "=", 800),
            ("wavelet.n_scales", "=", 200),
            ("spectral.use_welch", "=", True),
            ("spectral.nperseg", "*", 1.2),
            # Note: spectral.noverlap rule removed - standard 50% overlap optimal (Welch, 1967)
        ],
        DataLengthCategory.HUGE: [
            ("acf.max_lags_ratio", "=", 0.05),
            ("acf.nlags_limit", "=", 1000),
            ("wavelet.n_scales", "=", 300),
            ("spectral.nperseg", "*", 1.5),
            # Note: spectral.noverlap rule removed - standard 50% overlap optimal (Welch, 1967)
            ("spectral.nfft", "=", 1024),
        ],
        DataLengthCategory.MASSIVE: [
            ("base.max_missing_ratio", "*", 0.8),
            ("acf.max_lags_ratio", "=", 0.02),
            ("acf.nlags_limit", "=", 1500),
            ("wavelet.n_scales", "=", 500),
            ("wavelet.scale_to_period_factor", "=", 1.0),
            ("spectral.nperseg", "*", 2.0),
            # Note: spectral.noverlap rule removed - standard 50% overlap optimal (Welch, 1967)
            ("spectral.nfft", "=", 2048),
        ],
    },
    "instrument": {
        InstrumentTypeConfig.CRYPTO: [
            ("base.min_data_length", "*", 1.2),
            ("base.max_missing_ratio", "*", 0.9),
            ("acf.confidence_threshold", "*", 0.8),
            ("acf.peak_prominence", "*", 0.8),
            ("acf.nlags_limit", "*", 1.2),
            ("acf.bias_correction", "=", True),
            ("spectral.confidence_threshold", "*", 0.8),
            ("spectral.peak_height_ratio", "*", 0.8),
            ("spectral.n_peaks", "*", 1.5),
            ("spectral.use_welch", "=", True),
            ("spectral.scaling", "=", "density"),
            ("spectral.nperseg", "*", 0.8),
            # Note: spectral.noverlap rule removed - standard 50% overlap optimal (Welch, 1967)
            ("wavelet.confidence_threshold", "*", 0.8),
            ("wavelet.power_threshold", "*", 0.8),
            ("wavelet.n_peaks", "*", 1.5),
            ("wavelet.min_prominence", "*", 0.8),
            ("wavelet.use_cone_of_influence", "=", True),
        ],
    },
}

# Wavelet scale-to-period conversion factors (Torrence & Compo, 1998)
# Formula: period = scale × factor × sampling_period
WAVELET_FACTORS = {
    "cmor1.5-1.0": 1.0,   # Complex Morlet with fc=1.5
    "morl": 1.0,          # Morlet wavelet
    "mexh": np.sqrt(5/2), # Mexican Hat: sqrt(5/2)
    "cgau5": np.sqrt(5),  # Complex Gaussian: sqrt(5)
    "fbsp2-1-1.5": 1.0,   # Frequency B-Spline ⚠️ Requires verification
}

ADAPTIVE_RULES = {
    "volatility": {
        # High volatility adjustments (Tsay, 2010: stricter thresholds to reduce false positives)
        # Multiplier 1.5 increases thresholds by 50% to avoid spurious correlations
        # Example: confidence_threshold 0.3 → 0.45 (reduces false detection rate)
        "high": [
            ("acf.confidence_threshold", "*", 1.5),
            ("acf.correlation_threshold", "*", 1.5),
            ("acf.peak_prominence", "*", 1.5),
            ("spectral.confidence_threshold", "*", 1.5),
            ("spectral.peak_height_ratio", "*", 1.5),
            ("wavelet.confidence_threshold", "*", 1.5),
            ("wavelet.power_threshold", "*", 1.5),
        ],
        # Extreme volatility adjustments (Tsay, 2010: much stricter thresholds)
        # Multiplier 2.0 doubles thresholds to prevent spurious periodicity detection
        # Example: confidence_threshold 0.3 → 0.6 (critical for crypto trading)
        "extreme": [
            ("acf.confidence_threshold", "*", 2.0),
            ("acf.correlation_threshold", "*", 2.0),
            ("acf.peak_prominence", "*", 2.0),
            ("spectral.confidence_threshold", "*", 2.0),
            ("spectral.peak_height_ratio", "*", 2.0),
            ("wavelet.confidence_threshold", "*", 2.0),
            ("wavelet.power_threshold", "*", 2.0),
        ],
    },
    "stationarity": {
        "non_stationary": [
            ("spectral.detrend", "=", True),
            ("_weight.wavelet", "*", 1.2),
            ("_weight.acf", "*", 0.9),
        ],
        "stationary": [
            ("_weight.acf", "*", 1.1),
        ],
    },
    "noise": {
        "high": [
            ("acf.n_peaks", "*", 2),
            ("spectral.n_peaks", "*", 2),
            ("wavelet.n_peaks", "*", 2),
            ("_weight.wavelet", "*", 1.1),
        ],
    },
    "trend": {
        "strong": [
            ("spectral.detrend", "=", True),
            ("_weight.acf", "*", 0.8),
        ],
    },
}

RESEARCH_CONFIG = {
    "acf": {
        "min_period": 2,
        "max_period": 1000,
        "confidence_threshold": 0.05,
        "peak_prominence": 0.02,
        "correlation_threshold": 0.05,
        "max_lags_ratio": 0.8,
        "use_fft": True,
        "min_peak_distance": 1,
    },
    "spectral": {
        "min_period": 2,
        "max_period": 1000,
        "confidence_threshold": 0.05,
        "peak_height_ratio": 0.02,
        "window_type": "blackman",
        "detrend": True,
        "n_peaks": 20,
        "min_peak_distance": 1,
    },
    "wavelet": {
        "min_period": 2,
        "max_period": 1000,
        "confidence_threshold": 0.05,
        "wavelet": "cmor1.5-1.0",
        "n_scales": 500,
        "scale_distribution": "log",
        "power_threshold": 0.05,
        "n_peaks": 20,
    },
    "_active_methods": ["acf", "spectral", "wavelet"],
    "_weights": {"acf": 1.0, "spectral": 1.0, "wavelet": 1.0},
}


class PeriodicityConfigAdapter(BaseConfigAdapter):
    """
    Configuration adapter for periodicity detector.

    CRITICAL SEQUENCE in _pre_constraint_setup:
    1. _calculate_max_periods - sets max_period (needed for constraints)
    2. _calculate_spectral_parameters - calculate nperseg/noverlap/nfft
    """

    BASE: ClassVar[Dict[str, Dict[str, Any]]] = BASE
    ACTIVE: ClassVar[Dict[InstrumentTypeConfig, List[str]]] = ACTIVE
    RULES: ClassVar[Dict[str, Dict[Any, List[Tuple[str, str, Any]]]]] = RULES

    def _initialize_active_configs(
        self, classifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Dynamic initialization via STRUCTURES.

        SPECIFICITY: active methods change by frequency/length.
        """
        instrument = classifications["instrument"]
        frequency = classifications["frequency"]
        length_cat = classifications["length"]

        if instrument not in self.ACTIVE:
            raise ValueError(f"Instrument type {instrument.value} is not supported")

        structure = DEFAULT_STRUCTURE.copy()

        if frequency in STRUCTURES["frequency"]:
            self._merge_structures(structure, STRUCTURES["frequency"][frequency])

        if length_cat in STRUCTURES["length"]:
            self._merge_structures(structure, STRUCTURES["length"][length_cat])

        final_active_methods = structure.get("_active_methods", self.ACTIVE[instrument])

        config: Dict[str, Any] = {}
        for method in final_active_methods:
            if method in self.BASE:
                config[method] = deepcopy(self.BASE[method])

        config["_active_methods"] = final_active_methods
        config["_weights"] = structure.get("_weights", structure["weights"])

        for key, value in structure.items():
            if key not in ["methods", "weights"]:
                config[f"_{key}"] = value

        return config

    def _get_integer_parameter_map(self) -> Dict[str, List[str]]:
        """Integer parameter map."""
        return {
            "base": ["min_data_length"],
            "acf": ["min_period", "max_period", "min_peak_distance", "nlags_limit"],
            "spectral": [
                "min_period",
                "max_period",
                "n_peaks",
                "min_peak_distance",
                "nperseg",
                "noverlap",
            ],
            "wavelet": ["min_period", "max_period", "n_scales", "n_peaks"],
        }

    @staticmethod
    def _calculate_acf_confidence_threshold(
        data_length: int, alpha: float = 0.05
    ) -> float:
        """
        Calculate adaptive ACF confidence threshold based on data length.

        Mathematical rationale (Box & Jenkins, 1976, Section 2.1.6):
        For n-point time series, critical value for sample ACF at significance level α:
            threshold = z_critical / √n

        Where z_critical is the standard normal quantile (1.96 for α=0.05).

        This ensures consistent statistical power across different data lengths:
        - Short series (n=50): threshold ≈ 0.277 (less strict, avoid Type II errors)
        - Medium series (n=100): threshold ≈ 0.196 (balanced)
        - Long series (n=1000): threshold ≈ 0.062 (more strict, avoid Type I errors)

        Args:
            data_length: Number of observations in time series
            alpha: Significance level (default 0.05 for 95% confidence)

        Returns:
            Adaptive confidence threshold clamped to [0.05, 0.5] range

        Reference:
            Box, G.E.P. & Jenkins, G.M. (1976). Time Series Analysis:
            Forecasting and Control, Section 2.1.6
        """
        # Calculate z-critical value for two-tailed test
        z_critical = norm.ppf(1 - alpha / 2)  # ≈ 1.96 for α=0.05

        # Adaptive threshold: inversely proportional to √n
        threshold = z_critical / np.sqrt(data_length)

        # Clamp to reasonable range [0.05, 0.5]
        # Lower bound: prevent overly strict threshold for very long series
        # Upper bound: prevent overly loose threshold for very short series
        return max(0.05, min(0.5, threshold))

    def _pre_constraint_setup(self, config: Dict[str, Any], data_length: int) -> None:
        """
        CRITICAL SEQUENCE:
        1. max_period MUST be set FIRST (needed for constraints)
        2. Spectral parameters second
        3. Adaptive confidence thresholds third
        """
        # FIRST: set max_period for all methods
        self._calculate_max_periods(config, data_length)

        # SECOND: calculate spectral parameters
        self._calculate_spectral_parameters(config, data_length)

        # THIRD: set adaptive confidence thresholds
        # Apply to ACF, spectral, and wavelet methods
        adaptive_threshold = self._calculate_acf_confidence_threshold(data_length)

        if "acf" in config:
            config["acf"]["confidence_threshold"] = adaptive_threshold
            logging.debug(
                f"ACF adaptive threshold for n={data_length}: {adaptive_threshold:.4f}"
            )

        if "spectral" in config:
            config["spectral"]["confidence_threshold"] = adaptive_threshold
            logging.debug(
                f"Spectral adaptive threshold for n={data_length}: {adaptive_threshold:.4f}"
            )

        if "wavelet" in config:
            config["wavelet"]["confidence_threshold"] = adaptive_threshold
            logging.debug(
                f"Wavelet adaptive threshold for n={data_length}: {adaptive_threshold:.4f}"
            )

    def _apply_module_specific_constraints(
        self, config: Dict[str, Any], data_length: int
    ) -> None:
        """Mathematical constraints for periodicity."""
        # ACF constraints
        if "acf" in config:
            acf_config = config["acf"]
            theoretical_max_lags = data_length // 3

            if "max_lags_ratio" in acf_config:
                calculated_max_lags = int(acf_config["max_lags_ratio"] * data_length)
                if calculated_max_lags > theoretical_max_lags:
                    old_ratio = acf_config["max_lags_ratio"]
                    acf_config["max_lags_ratio"] = theoretical_max_lags / data_length
                    logging.debug(
                        f"ACF max_lags_ratio constrained: {old_ratio:.3f} -> "
                        f"{acf_config['max_lags_ratio']:.3f}"
                    )

            if "nlags_limit" in acf_config:
                max_allowed_nlags = min(theoretical_max_lags, data_length // 2)
                if acf_config["nlags_limit"] > max_allowed_nlags:
                    logging.debug(
                        f"ACF nlags_limit constrained: {acf_config['nlags_limit']} -> "
                        f"{max_allowed_nlags}"
                    )
                    acf_config["nlags_limit"] = max(max_allowed_nlags, 10)

            # SAFE max_period comparison (already set in _pre_constraint_setup)
            if "max_period" in acf_config and acf_config["max_period"] is not None:
                max_allowed_period = min(theoretical_max_lags, data_length // 2)
                if acf_config["max_period"] > max_allowed_period:
                    logging.debug(
                        f"ACF max_period constrained: {acf_config['max_period']} -> "
                        f"{max_allowed_period}"
                    )
                    acf_config["max_period"] = max_allowed_period

        # Spectral constraints
        if "spectral" in config:
            spectral_config = config["spectral"]
            theoretical_max_nperseg = data_length // 2

            if "nperseg" in spectral_config and spectral_config["nperseg"] is not None:
                if spectral_config["nperseg"] > theoretical_max_nperseg:
                    logging.debug(
                        f"Spectral nperseg constrained: {spectral_config['nperseg']} -> "
                        f"{theoretical_max_nperseg}"
                    )
                    spectral_config["nperseg"] = max(theoretical_max_nperseg, 8)

            # Enforce optimal 50% overlap (Welch, 1967)
            if "nperseg" in spectral_config and spectral_config["nperseg"] is not None:
                optimal_overlap = spectral_config["nperseg"] // 2

                if "noverlap" in spectral_config and spectral_config["noverlap"] is not None:
                    # Check if noverlap deviates from optimal 50%
                    if spectral_config["noverlap"] != optimal_overlap:
                        logging.info(
                            f"Enforcing optimal Welch overlap (50%): "
                            f"{spectral_config['noverlap']} → {optimal_overlap} "
                            f"(nperseg={spectral_config['nperseg']}, reference: Welch 1967)"
                        )
                        spectral_config["noverlap"] = optimal_overlap
                else:
                    # Set noverlap if not specified
                    spectral_config["noverlap"] = optimal_overlap
                    logging.debug(
                        f"Spectral noverlap set to optimal 50%: {optimal_overlap}"
                    )

            if "nfft" in spectral_config and spectral_config["nfft"] is not None:
                max_allowed_nfft = data_length * 2
                if spectral_config["nfft"] > max_allowed_nfft:
                    logging.debug(
                        f"Spectral nfft constrained: {spectral_config['nfft']} -> "
                        f"{max_allowed_nfft}"
                    )
                    spectral_config["nfft"] = max_allowed_nfft

            # SAFE comparison
            if (
                "max_period" in spectral_config
                and spectral_config["max_period"] is not None
            ):
                max_allowed_period = min(data_length // 2, 1000)
                if spectral_config["max_period"] > max_allowed_period:
                    logging.debug(
                        f"Spectral max_period constrained: {spectral_config['max_period']} -> "
                        f"{max_allowed_period}"
                    )
                    spectral_config["max_period"] = max_allowed_period

        # Wavelet constraints
        if "wavelet" in config:
            wavelet_config = config["wavelet"]
            theoretical_max_scales = data_length // 4

            if "n_scales" in wavelet_config:
                if wavelet_config["n_scales"] > theoretical_max_scales:
                    logging.debug(
                        f"Wavelet n_scales constrained: {wavelet_config['n_scales']} -> "
                        f"{theoretical_max_scales}"
                    )
                    wavelet_config["n_scales"] = max(theoretical_max_scales, 10)

            if "sampling_period" in wavelet_config:
                max_allowed_sampling = data_length / 10.0
                if wavelet_config["sampling_period"] > max_allowed_sampling:
                    logging.debug(
                        f"Wavelet sampling_period constrained: "
                        f"{wavelet_config['sampling_period']:.3f} -> "
                        f"{max_allowed_sampling:.3f}"
                    )
                    wavelet_config["sampling_period"] = max(max_allowed_sampling, 0.1)

            if "scale_to_period_factor" in wavelet_config and data_length < 100:
                if wavelet_config["scale_to_period_factor"] > 2.0:
                    logging.debug(
                        f"Wavelet scale_to_period_factor constrained for short data: "
                        f"{wavelet_config['scale_to_period_factor']} -> 2.0"
                    )
                    wavelet_config["scale_to_period_factor"] = 2.0

            # SAFE comparison
            if (
                "max_period" in wavelet_config
                and wavelet_config["max_period"] is not None
            ):
                max_allowed_period = min(data_length // 4, 500)
                if wavelet_config["max_period"] > max_allowed_period:
                    logging.debug(
                        f"Wavelet max_period constrained: {wavelet_config['max_period']} -> "
                        f"{max_allowed_period}"
                    )
                    wavelet_config["max_period"] = max_allowed_period

    def _validate_module_specific_ranges(self, config: Dict[str, Any]) -> None:
        """Range validation with automatic correction."""
        # ACF validation
        if "acf" in config:
            acf_config = config["acf"]

            for threshold_param in ["confidence_threshold", "correlation_threshold"]:
                if threshold_param in acf_config:
                    value = acf_config[threshold_param]
                    if not (0 < value < 1):
                        clamped = max(0.001, min(0.999, value))
                        logging.warning(
                            f"ACF {threshold_param} out of range (0,1): "
                            f"{value} -> {clamped}"
                        )
                        acf_config[threshold_param] = clamped

            if "peak_prominence" in acf_config:
                value = acf_config["peak_prominence"]
                if value <= 0:
                    clamped = 0.05
                    logging.warning(
                        f"ACF peak_prominence too small: {value} -> {clamped}"
                    )
                    acf_config["peak_prominence"] = clamped
                elif value > 1.0:
                    clamped = 1.0
                    logging.warning(
                        f"ACF peak_prominence too large: {value} -> {clamped}"
                    )
                    acf_config["peak_prominence"] = clamped

            if "max_lags_ratio" in acf_config:
                value = acf_config["max_lags_ratio"]
                if value > 0.5:
                    clamped = 0.5
                    logging.warning(
                        f"ACF max_lags_ratio too large: {value} -> {clamped}"
                    )
                    acf_config["max_lags_ratio"] = clamped
                elif value <= 0:
                    clamped = 0.1
                    logging.warning(
                        f"ACF max_lags_ratio too small: {value} -> {clamped}"
                    )
                    acf_config["max_lags_ratio"] = clamped

            if "nlags_limit" in acf_config:
                value = acf_config["nlags_limit"]
                if value <= 0:
                    clamped = 100
                    logging.warning(
                        f"ACF nlags_limit too small: {value} -> {clamped}"
                    )
                    acf_config["nlags_limit"] = clamped
                elif value > 5000:
                    clamped = 5000
                    logging.warning(
                        f"ACF nlags_limit too large: {value} -> {clamped}"
                    )
                    acf_config["nlags_limit"] = clamped

            if "bias_correction" in acf_config:
                value = acf_config["bias_correction"]
                if not isinstance(value, bool):
                    clamped = True
                    logging.warning(
                        f"ACF bias_correction not bool: {value} -> {clamped}"
                    )
                    acf_config["bias_correction"] = clamped

        # Spectral validation
        if "spectral" in config:
            spectral_config = config["spectral"]

            for threshold_param in ["confidence_threshold", "peak_height_ratio"]:
                if threshold_param in spectral_config:
                    value = spectral_config[threshold_param]
                    if not (0 < value < 1):
                        clamped = max(0.001, min(0.999, value))
                        logging.warning(
                            f"Spectral {threshold_param} out of range (0,1): "
                            f"{value} -> {clamped}"
                        )
                        spectral_config[threshold_param] = clamped

            if "n_peaks" in spectral_config:
                value = spectral_config["n_peaks"]
                if value <= 0:
                    clamped = 1
                    logging.warning(
                        f"Spectral n_peaks too small: {value} -> {clamped}"
                    )
                    spectral_config["n_peaks"] = clamped
                elif value > 50:
                    clamped = 50
                    logging.warning(
                        f"Spectral n_peaks too large: {value} -> {clamped}"
                    )
                    spectral_config["n_peaks"] = clamped

            if "window_type" in spectral_config:
                valid_windows = ["hann", "hamming", "blackman", "bartlett", "boxcar"]
                if spectral_config["window_type"] not in valid_windows:
                    default_window = "hann"
                    logging.warning(
                        f"Spectral unknown window_type: "
                        f"{spectral_config['window_type']} -> {default_window}"
                    )
                    spectral_config["window_type"] = default_window

            if "scaling" in spectral_config:
                valid_scalings = ["density", "spectrum"]
                if spectral_config["scaling"] not in valid_scalings:
                    default_scaling = "density"
                    logging.warning(
                        f"Spectral unknown scaling: "
                        f"{spectral_config['scaling']} -> {default_scaling}"
                    )
                    spectral_config["scaling"] = default_scaling

            if "use_welch" in spectral_config:
                value = spectral_config["use_welch"]
                if not isinstance(value, bool):
                    clamped = True
                    logging.warning(f"Spectral use_welch not bool: {value} -> {clamped}")
                    spectral_config["use_welch"] = clamped

            for param in ["nperseg", "noverlap", "nfft"]:
                if param in spectral_config:
                    value = spectral_config[param]
                    if value is not None and value <= 0:
                        logging.warning(
                            f"Spectral {param} too small: {value} -> None (auto-select)"
                        )
                        spectral_config[param] = None

        # Wavelet validation
        if "wavelet" in config:
            wavelet_config = config["wavelet"]

            for threshold_param in ["confidence_threshold", "power_threshold"]:
                if threshold_param in wavelet_config:
                    value = wavelet_config[threshold_param]
                    if not (0 < value < 1):
                        clamped = max(0.001, min(0.999, value))
                        logging.warning(
                            f"Wavelet {threshold_param} out of range (0,1): "
                            f"{value} -> {clamped}"
                        )
                        wavelet_config[threshold_param] = clamped

            if "n_scales" in wavelet_config:
                value = wavelet_config["n_scales"]
                if value <= 0:
                    clamped = 10
                    logging.warning(
                        f"Wavelet n_scales too small: {value} -> {clamped}"
                    )
                    wavelet_config["n_scales"] = clamped
                elif value > 1000:
                    clamped = 1000
                    logging.warning(
                        f"Wavelet n_scales too large: {value} -> {clamped}"
                    )
                    wavelet_config["n_scales"] = clamped

            if "n_peaks" in wavelet_config:
                value = wavelet_config["n_peaks"]
                if value <= 0:
                    clamped = 1
                    logging.warning(
                        f"Wavelet n_peaks too small: {value} -> {clamped}"
                    )
                    wavelet_config["n_peaks"] = clamped
                elif value > 50:
                    clamped = 50
                    logging.warning(
                        f"Wavelet n_peaks too large: {value} -> {clamped}"
                    )
                    wavelet_config["n_peaks"] = clamped

            if "scale_distribution" in wavelet_config:
                valid_distributions = ["log", "linear"]
                if wavelet_config["scale_distribution"] not in valid_distributions:
                    default_distribution = "log"
                    logging.warning(
                        f"Wavelet unknown scale_distribution: "
                        f"{wavelet_config['scale_distribution']} -> {default_distribution}"
                    )
                    wavelet_config["scale_distribution"] = default_distribution

            if "wavelet" in wavelet_config:
                valid_wavelets = [
                    "cmor1.5-1.0",
                    "morl",
                    "mexh",
                    "cgau5",
                    "fbsp2-1-1.5",
                ]
                if wavelet_config["wavelet"] not in valid_wavelets:
                    default_wavelet = "cmor1.5-1.0"
                    logging.warning(
                        f"Wavelet unknown wavelet: "
                        f"{wavelet_config['wavelet']} -> {default_wavelet}"
                    )
                    wavelet_config["wavelet"] = default_wavelet

            if "sampling_period" in wavelet_config:
                value = wavelet_config["sampling_period"]
                if value <= 0:
                    clamped = 1.0
                    logging.warning(
                        f"Wavelet sampling_period too small: {value} -> {clamped}"
                    )
                    wavelet_config["sampling_period"] = clamped

            if "min_prominence" in wavelet_config:
                value = wavelet_config["min_prominence"]
                if not (0 <= value <= 1):
                    clamped = max(0.0, min(1.0, value))
                    logging.warning(
                        f"Wavelet min_prominence out of range [0,1]: "
                        f"{value} -> {clamped}"
                    )
                    wavelet_config["min_prominence"] = clamped

            if "use_cone_of_influence" in wavelet_config:
                value = wavelet_config["use_cone_of_influence"]
                if not isinstance(value, bool):
                    clamped = True
                    logging.warning(
                        f"Wavelet use_cone_of_influence not bool: {value} -> {clamped}"
                    )
                    wavelet_config["use_cone_of_influence"] = clamped

            if "scale_to_period_factor" in wavelet_config:
                value = wavelet_config["scale_to_period_factor"]
                if value <= 0:
                    clamped = 1.0
                    logging.warning(
                        f"Wavelet scale_to_period_factor too small: "
                        f"{value} -> {clamped}"
                    )
                    wavelet_config["scale_to_period_factor"] = clamped
                elif value > 10.0:
                    clamped = 10.0
                    logging.warning(
                        f"Wavelet scale_to_period_factor too large: "
                        f"{value} -> {clamped}"
                    )
                    wavelet_config["scale_to_period_factor"] = clamped

            # NEW CODE: Validate wavelet factor consistency (Torrence & Compo, 1998)
            if "wavelet" in wavelet_config and "scale_to_period_factor" in wavelet_config:
                wavelet_type = wavelet_config["wavelet"]

                if wavelet_type in WAVELET_FACTORS:
                    expected_factor = WAVELET_FACTORS[wavelet_type]
                    actual_factor = wavelet_config["scale_to_period_factor"]

                    # Tolerance 10% for numerical precision
                    if abs(actual_factor - expected_factor) > 0.1:
                        logging.warning(
                            f"Wavelet scale_to_period_factor mismatch detected: "
                            f"actual={actual_factor:.2f} ≠ expected={expected_factor:.2f} "
                            f"for wavelet type '{wavelet_type}'. "
                            f"Auto-correcting to canonical value (Torrence & Compo, 1998)."
                        )
                        wavelet_config["scale_to_period_factor"] = expected_factor
                else:
                    # Unknown wavelet type - log warning but don't change
                    logging.info(
                        f"Unknown wavelet type '{wavelet_type}' - cannot validate "
                        f"scale_to_period_factor. Using configured value: "
                        f"{wavelet_config['scale_to_period_factor']}"
                    )

        # Common validation
        for method in config:
            if method.startswith("_"):
                continue
            method_config = config[method]

            if "min_period" in method_config and "max_period" in method_config:
                min_period = method_config["min_period"]
                max_period = method_config["max_period"]

                if min_period <= 0:
                    method_config["min_period"] = 2
                    logging.warning(
                        f"{method} min_period corrected: {min_period} -> 2"
                    )

                # SAFE max_period check
                if max_period is not None and max_period <= min_period:
                    method_config["max_period"] = min_period + 1
                    logging.warning(
                        f"{method} max_period corrected: "
                        f"{max_period} -> {min_period + 1}"
                    )

    def _apply_crypto_adjustments(
        self, config: Dict[str, Any], volatility: Optional[float] = None
    ) -> None:
        """Cryptocurrency adjustments."""
        if volatility is not None:
            volatility_level = self._classify_crypto_periodicity_volatility(volatility)

            if volatility_level in ["variable", "chaotic"]:
                logging.info(
                    f"Applying adjustments for {volatility_level} "
                    f"periodicity volatility: {volatility:.3f}"
                )

                multiplier = 1.5 if volatility_level == "chaotic" else 1.2

                for (
                    param_path,
                    adjustment,
                ) in CRYPTO_HIGH_VOLATILITY_ADJUSTMENTS.items():
                    method, param = param_path.split(".")

                    if method in config and param in config[method]:
                        old_value = config[method][param]

                        if isinstance(adjustment, bool):
                            new_value = adjustment
                        elif isinstance(adjustment, str):
                            new_value = adjustment
                        elif isinstance(adjustment, (int, float)) and isinstance(
                            old_value, (int, float)
                        ):
                            new_value = old_value * (adjustment * multiplier)
                        else:
                            continue

                        config[method][param] = new_value

                        logging.debug(
                            f"Volatility {volatility_level}: "
                            f"{param_path} {old_value} -> {new_value}"
                        )
        else:
            logging.debug(
                "Applying base crypto adjustments for periodicity "
                "(volatility unknown)"
            )

            for param_path, adjustment in CRYPTO_HIGH_VOLATILITY_ADJUSTMENTS.items():
                method, param = param_path.split(".")

                if method in config and param in config[method]:
                    old_value = config[method][param]

                    if isinstance(adjustment, bool):
                        new_value = adjustment
                    elif isinstance(adjustment, str):
                        new_value = adjustment
                    elif isinstance(adjustment, (int, float)) and isinstance(
                        old_value, (int, float)
                    ):
                        new_value = old_value * min(adjustment, 1.1)
                    else:
                        continue

                    config[method][param] = new_value

    def _apply_module_specific_adaptations(
        self,
        config: Dict[str, Any],
        params: Dict[str, Any],
        classifications: Dict[str, Any],
    ) -> None:
        """Adaptive rules."""
        adaptations = []

        if "volatility" in params and params["volatility"] is not None:
            if params["volatility"] > 0.9:
                for rule in ADAPTIVE_RULES["volatility"]["extreme"]:
                    self._apply_single_rule(config, rule)
                adaptations.append(f"volatility=extreme({params['volatility']:.2f})")
            elif params["volatility"] > 0.7:
                for rule in ADAPTIVE_RULES["volatility"]["high"]:
                    self._apply_single_rule(config, rule)
                adaptations.append(f"volatility=high({params['volatility']:.2f})")

        if "stationarity" in params and params["stationarity"] is not None:
            if params["stationarity"]:
                for rule in ADAPTIVE_RULES["stationarity"]["stationary"]:
                    self._apply_single_rule(config, rule)
                adaptations.append("stationarity=True")
            else:
                for rule in ADAPTIVE_RULES["stationarity"]["non_stationary"]:
                    self._apply_single_rule(config, rule)
                adaptations.append("stationarity=False")

        if "noise_level" in params and params["noise_level"] is not None:
            if params["noise_level"] > 0.7:
                for rule in ADAPTIVE_RULES["noise"]["high"]:
                    self._apply_single_rule(config, rule)
                adaptations.append(f"noise=high({params['noise_level']:.2f})")

        if (
            "estimated_trend_strength" in params
            and params["estimated_trend_strength"] is not None
        ):
            if params["estimated_trend_strength"] > 0.7:
                for rule in ADAPTIVE_RULES["trend"]["strong"]:
                    self._apply_single_rule(config, rule)
                adaptations.append(
                    f"trend=strong({params['estimated_trend_strength']:.2f})"
                )

        if adaptations:
            logging.info(f"Applied adaptive rules: {', '.join(adaptations)}")

    def _finalize_module_specific(
        self, config: Dict[str, Any], params: Dict[str, Any]
    ) -> None:
        """
        Finalization: normalize weights and limit n_peaks.

        NOTE: max_period is ALREADY set in _pre_constraint_setup.
        """
        # Normalize weights
        self._normalize_weights(config)

        # Limit n_peaks
        for method in config.get("_active_methods", []):
            if method in config and "n_peaks" in config[method]:
                config[method]["n_peaks"] = min(int(config[method]["n_peaks"]), 30)

    def _transform_rule_value(self, method: str, param: str, value: Any) -> Any:
        """Transform spectral parameters to int."""
        if method == "spectral" and param in ["nperseg", "noverlap", "nfft"]:
            if value is not None:
                return int(value)
        return value

    # ========== HELPER METHODS ==========

    def _merge_structures(self, base: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Merge structural changes."""
        for key, value in updates.items():
            if key == "methods":
                base["_active_methods"] = value
            elif key == "weights":
                base["_weights"] = {**base.get("_weights", {}), **value}
            else:
                base[f"_{key}"] = value

    def _calculate_max_periods(self, config: Dict[str, Any], data_length: int) -> None:
        """
        Calculate max_period for all methods using scientifically justified constraints.

        Mathematical rationale (Box & Jenkins, 1976):
        - Minimum 3 full cycles required for statistical significance
        - Nyquist limit: max_period ≤ data_length // 2
        - Computational efficiency caps for long series

        Formula: max_period = min(nyquist_limit, data_length // min_cycles, cap)

        Examples:
        - data_length=50:  max_period=16  (3 cycles minimum, Nyquist=25)
        - data_length=300: max_period=100 (3 cycles minimum, capped at 300)
        - data_length=2000: max_period=500 (4 cycles, computational cap)

        Reference:
            Box, G.E.P. & Jenkins, G.M. (1976). Time Series Analysis:
            Forecasting and Control, Section 2.1.6

        Args:
            config: Configuration dictionary to update
            data_length: Length of time series data

        CRITICAL: Must be called BEFORE applying constraints!
        """
        # Nyquist limit: max_period cannot exceed half of data length
        nyquist_limit = data_length // 2

        # Apply 3-cycle minimum requirement with data-dependent caps
        if data_length < 100:
            # Short series: prioritize Nyquist limit and 3-cycle requirement
            max_period = min(nyquist_limit, data_length // 3)
        elif data_length < 1000:
            # Medium series: 3-cycle requirement with computational cap
            max_period = min(data_length // 3, 300)
        else:
            # Long series: 4-cycle for efficiency with increased cap
            max_period = min(data_length // 4, 500)

        for method in config.get("_active_methods", []):
            if method in config:
                config[method]["max_period"] = max(
                    max_period, config[method]["min_period"] + 1
                )

    def _normalize_weights(self, config: Dict[str, Any]) -> None:
        """
        Normalize method weights with comprehensive numerical stability.

        Handles edge cases:
        - Negative weights: clamped to 0 and renormalized
        - Zero total: fallback to equal weights (1/n)
        - Near-zero total: warning logged and equal weights used
        - No rounding: preserves full precision to ensure sum == 1.0

        Mathematical rationale:
        - Weights must be non-negative (physical constraint)
        - Sum must equal 1.0 for proper consensus voting
        - Numerical tolerance 1e-10 for zero detection
        - Assertion ensures sum == 1.0 within 1e-6 tolerance

        Args:
            config: Configuration dictionary containing _weights

        Raises:
            AssertionError: If normalized weights sum != 1.0 (tolerance 1e-6)
        """
        if "_weights" not in config:
            return

        weights = config["_weights"]
        n_methods = len(weights)

        if n_methods == 0:
            return

        # Step 1: Clamp negative weights to 0
        has_negative = False
        for key, value in weights.items():
            if value < 0:
                has_negative = True
                logging.warning(
                    f"Negative weight detected: {key}={value:.6f} → clamped to 0"
                )
                weights[key] = 0.0

        # Step 2: Calculate total with numerical tolerance
        total = sum(weights.values())
        zero_tolerance = 1e-10

        # Step 3: Handle zero or near-zero total
        if abs(total) < zero_tolerance:
            # Fallback to equal weights
            equal_weight = 1.0 / n_methods
            if total < 1e-8:
                logging.warning(
                    f"Weight total near zero: {total:.2e} → using equal weights "
                    f"({equal_weight:.6f} each)"
                )
            config["_weights"] = {k: equal_weight for k in weights.keys()}
        else:
            # Step 4: Normalize weights (no rounding to preserve sum == 1.0)
            config["_weights"] = {k: v / total for k, v in weights.items()}

            if has_negative:
                logging.info(
                    f"Weights renormalized after clamping negative values: "
                    f"sum={sum(config['_weights'].values()):.6f}"
                )

        # Step 5: Assertion to ensure sum == 1.0
        final_sum = sum(config["_weights"].values())
        sum_tolerance = 1e-6
        assert abs(final_sum - 1.0) < sum_tolerance, (
            f"Weight normalization failed: sum={final_sum:.10f} != 1.0 "
            f"(tolerance={sum_tolerance})"
        )

    def _calculate_spectral_parameters(
        self, config: Dict[str, Any], data_length: int
    ) -> None:
        """
        Calculate FFT-optimized spectral parameters for Welch method.

        Mathematical rationale:
        - nperseg as power-of-2 ensures O(n log n) FFT complexity
        - Minimum nperseg=32 (2^5) for acceptable frequency resolution
        - Standard 50% overlap (Welch, 1967) for optimal variance reduction
        - nfft zero-padding improves interpolation accuracy

        Formula:
            nperseg = 2^max(5, floor(log2(data_length/4)))
            nperseg = min(nperseg, data_length//2, 512)
            noverlap = nperseg // 2  (standard 50%)
            nfft = 2^(floor(log2(nperseg)) + 1)  if data_length > 1000

        Examples:
        - data_length=32:   nperseg=32 (2^5), noverlap=16, nfft=None
        - data_length=256:  nperseg=64 (2^6), noverlap=32, nfft=None
        - data_length=2000: nperseg=256 (2^8), noverlap=128, nfft=512 (2^9)

        References:
            Welch, P.D. (1967). The use of fast Fourier transform for the
                estimation of power spectra. IEEE Transactions on Audio
                and Electroacoustics, 15(2), 70-73.
            Cooley, J.W., & Tukey, J.W. (1965). An algorithm for the
                machine calculation of complex Fourier series.
                Mathematics of Computation, 19(90), 297-301.

        Args:
            config: Configuration dictionary to update
            data_length: Length of time series data
        """
        if "spectral" not in config:
            return

        spectral_config = config["spectral"]

        if not spectral_config.get("use_welch", True):
            spectral_config["nperseg"] = None
            spectral_config["noverlap"] = None
            spectral_config["nfft"] = None
            return

        # Calculate nperseg as power-of-2 for FFT efficiency
        # Formula: nperseg = 2^max(5, floor(log2(data_length/4)))
        # This ensures minimum 32 and scales with data length
        min_power = 5  # Minimum 2^5 = 32 (Welch, 1967)
        target_power = max(min_power, int(math.log2(data_length / 4)))
        nperseg = 2**target_power

        # Apply upper constraints
        # - Cannot exceed data_length // 2 (Nyquist-related practical limit)
        # - Cap at 512 for computational efficiency
        nperseg = min(nperseg, data_length // 2, 512)

        # Ensure nperseg is at least min_power (32) and does not exceed data_length
        nperseg = max(2**min_power, min(nperseg, data_length))

        # Standard 50% overlap (Welch, 1967)
        # This is optimal for variance reduction without redundancy
        noverlap = nperseg // 2

        # Calculate nfft for long series (zero-padding improves interpolation)
        # nfft = next power of 2 after nperseg
        nfft = None
        if data_length > 1000:
            # Next power of 2: 2^(floor(log2(nperseg)) + 1)
            nfft_power = int(math.log2(nperseg)) + 1
            nfft = 2**nfft_power
            # Cap at 2048 for memory efficiency
            nfft = min(nfft, 2048)

        spectral_config["nperseg"] = int(nperseg)
        spectral_config["noverlap"] = int(noverlap)
        spectral_config["nfft"] = int(nfft) if nfft is not None else None

        logging.debug(
            f"FFT-optimized spectral parameters for data_length={data_length}: "
            f"nperseg={nperseg} (2^{int(math.log2(nperseg))}), "
            f"noverlap={noverlap}, nfft={nfft}"
        )

    @staticmethod
    def _classify_crypto_periodicity_volatility(volatility: float) -> str:
        """Classify cryptocurrency periodicity volatility."""
        if volatility <= CRYPTO_PERIODICITY_VOLATILITY_THRESHOLDS["stable"]:
            return "stable"
        elif volatility <= CRYPTO_PERIODICITY_VOLATILITY_THRESHOLDS["variable"]:
            return "variable"
        else:
            return "chaotic"


# ========== FACTORY FUNCTION ==========

_adapter = PeriodicityConfigAdapter()


def build_config_from_properties(params: Dict[str, Any]) -> Dict[str, Any]:
    """Factory function for backward compatibility."""
    return _adapter.build_config_from_properties(params)


def __str__() -> str:
    """String representation for diagnostics."""
    return (
        f"PeriodicityConfigAdapter(version={__version__}, methods={list(BASE.keys())})"
    )