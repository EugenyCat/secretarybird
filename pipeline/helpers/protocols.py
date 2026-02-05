from typing import Protocol
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional, Union

class ETLProtocol(Protocol):
    """
        pattern Abstract Factory
        ETLProtocol - ETL process interface from different sources
    """

    def validate_parameters(self, input_params: dict) -> (dict, int):
        """
            Validate required parameters
        """
        ...

    def extract_data(self) -> (dict or None, None or dict):
        """
            Extract data from source
        """
        ...

    def transform_data(self, extracted_data: list) -> (dict or None, None or dict):
        """
            Transform raw data into a Pandas DataFrame with the correct fieldnames and datatypes.
        """
        ...

    def load_data(self, transformed_data: list) -> (dict or None, None or dict):
        """
            Insert data into ClickHouse db
        """
        ...

    def run_etl(self, input_params: dict) -> (dict, dict):
        """
            Orchestrates the entire ETL process: extraction, transformation, and loading of data.
        """
        ...



class ConfigurationAdapterProtocol(Protocol):
    """
    Unified protocol for time series configuration adapters.

    Defines common structure and interface for creating adaptive configurations
    for various time series modules (Analyzer, Periodicity, Decomposition,
    OutlierRemover, Features).

    === ARCHITECTURAL PRINCIPLES ===

    1. **Structure uniformity**: All adapters use the same patterns:
       - BASE: base configurations with sections for methods
       - ACTIVE: active methods for instrument types
       - RULES: adaptation rules by categories (frequency, length, instrument)

    2. **Adaptability**: Configurations adapt to data characteristics:
       - Data frequency (HIGH/MEDIUM/LOW)
       - Data length (TINY/SHORT/SMALL/LARGE/HUGE/MASSIVE)
       - Instrument type (CRYPTO and others)

    3. **Safety**: Protection from incorrect parameters:
       - Range validation after applying rules
       - Parameter limiting by actual data length
       - Automatic correction with logging

    4. **Performance**: Intelligent optimization:
       - Limiting computationally expensive operations
       - Balancing accuracy and speed
       - Mathematical correctness of algorithms

    === STRUCTURAL ELEMENTS ===

    BASE: Dict[str, Dict[str, Any]]
        Base configurations for each module method.
        Example: {"base": {...}, "method1": {...}, "method2": {...}}

    ACTIVE: Dict[InstrumentTypeConfig, List[str]]
        Active methods for various instrument types.
        Example: {InstrumentTypeConfig.CRYPTO: ["base", "method1", "method2"]}

    RULES: Dict[str, Dict[Any, List[Tuple[str, str, Any]]]]
        Parameter adaptation rules by categories.
        Rule format: (parameter, operation, value)
        Operations: "*" (multiply), "+" (add), "=" (set)

    === OPERATION SEQUENCE ===

    1. Validate input parameters (validate_required_locals)
    2. Classify data (frequency, length, instrument)
    3. Initialize configurations for active methods
    4. Propagate base parameters to all methods
    5. Apply adaptation rules in order: frequency -> length -> instrument
    6. Constrain parameters by data length
    7. Validate parameter ranges
    8. Apply specific adjustments (crypto, volatility)
    9. Type conversion and rounding
    10. Return unified configuration

    === IMPLEMENTATION REQUIREMENTS ===

    - Compliance with SOLID, DRY, KISS principles
    - Detailed logging of all operations
    - Backward compatibility
    - Performance no more than 50ms per configuration
    - Mathematical correctness of parameters
    """

    # === MAIN FUNCTION ===

    def build_config_from_properties(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build configuration based on data properties.

        Args:
            params: Dict with required keys:
                - instrument_type: InstrumentTypeConfig
                - interval: str (data interval)
                - data_length: int (time series length)

                Optional keys (depend on module):
                - volatility: float (0.0-1.0)
                - stationarity: bool
                - noise_level: float (0.0-1.0)
                - trend_strength: float (0.0-1.0)

        Returns:
            Dict with configuration in unified format:
            {
                "method1": {...},  # Method configurations
                "method2": {...},
                "_active_methods": [...],  # List of active methods
                "_weights": {...},  # Method weights (if applicable)
                "_verify_with_stl": bool,  # Additional flags (if applicable)
                ...
            }
        """
        ...

    # === RULE APPLICATION PATTERNS ===

    def _apply_rule(self, config: Dict[str, Any], rule: Tuple[str, str, Any]) -> None:
        """
        Apply a single rule to configuration.

        Args:
            config: Configuration to modify
            rule: Rule in format (parameter, operation, value)
                 parameter: "method.param" (e.g., "acf.confidence_threshold")
                 operation: "*" (multiply), "+" (add), "=" (set)
                 value: number or other value for the operation

        Processing patterns:
        - Support for nested parameters (method.param)
        - Special handling of weights (_weight.method)
        - Safe handling of None values
        - Detailed logging of changes
        """
        ...

    # === VALIDATION AND CONSTRAINTS ===

    def _constrain_to_data(self, config: Dict[str, Any], data_length: int) -> None:
        """
        Constrain parameters by actual data length.

        Mathematical constraints by length:
        - Window sizes ≤ data_length/3
        - Number of lags ≤ data_length/4
        - Segments ≤ data_length/2
        - Scales ≤ data_length/4

        Args:
            config: Configuration to constrain
            data_length: Time series length
        """
        ...

    def _validate_ranges(self, config: Dict[str, Any]) -> None:
        """
        Validate parameter ranges with protection against boundary violations.

        Checked ranges:
        - alpha parameters ∈ (0, 1)
        - ratios ∈ [0, 1]
        - threshold > 0
        - counts ≥ 1
        - window_type ∈ valid_values

        When incorrect values are detected:
        - Automatic correction (clamping)
        - Warning logging
        - Maintain operability

        Args:
            config: Configuration to validate
        """
        ...

    # === SPECIFIC ADJUSTMENTS ===

    def _apply_crypto_adjustments(
            self, config: Dict[str, Any], instrument, volatility: Optional[float] = None
    ) -> None:
        """
        Apply specific adjustments for cryptocurrencies.

        Adaptations for high volatility:
        - More tolerant thresholds
        - Increased analysis windows
        - Less strict statistical tests
        - Estimate stabilization

        Args:
            config: Configuration to adjust
            instrument: Instrument type
            volatility: Volatility level (optional)
        """
        ...

    # === FINAL PROCESSING ===

    def _process_integer_and_round_params(self, config: dict) -> None:
        """
        Type conversion and rounding for readability.

        Conversions:
        - Counter parameters to int
        - Rounding float to 4 decimal places
        - Preserving logical structure

        Args:
            config: Configuration to process
        """
        ...

    # === STRUCTURAL CONSTANTS (must be defined in implementation) ===

    @property
    def BASE(self) -> Dict[str, Dict[str, Any]]:
        """Base configurations for all module methods."""
        ...

    @property
    def ACTIVE(self) -> Dict[Any, List[str]]:
        """Active methods for various instrument types."""
        ...

    @property
    def RULES(self) -> Dict[str, Dict[Any, List[Tuple[str, str, Any]]]]:
        """Parameter adaptation rules by categories."""
        ...


# === PROTOCOL USAGE EXAMPLE ===

"""
Configuration adapter implementation example:

class ConfigDecomposition:
    '''Configuration adapter for decomposition module.'''

    # Structural constants
    BASE = {
        "base": {"min_data_length": 10, "max_missing_ratio": 0.3},
        "stl": {"seasonal": 7, "trend": None, "robust": False},
        "prophet": {"seasonality_mode": "additive", "daily": True},
    }

    ACTIVE = {
        InstrumentTypeConfig.CRYPTO: ["base", "stl", "prophet"]
    }

    RULES = {
        "frequency": {
            FrequencyCategory.HIGH: [
                ("stl.seasonal", "*", 0.1),
                ("prophet.daily", "=", False),
            ]
        }
    }

    def build_config_from_properties(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Implementation according to protocol...
        return config

    def _apply_rule(self, config: Dict[str, Any], rule: Tuple[str, str, Any]) -> None:
        # Standard implementation...
        pass

    # Other protocol methods...

# Usage:
adapter = ConfigDecomposition()
config = adapter.build_config_from_properties({
    "instrument_type": InstrumentTypeConfig.CRYPTO,
    "interval": "1h", 
    "data_length": 1000
})
"""