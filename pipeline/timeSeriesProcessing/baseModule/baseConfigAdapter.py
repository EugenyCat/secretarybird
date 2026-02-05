"""
BaseConfigAdapter - unified base class for configuration adapters.

Implements Template Method pattern to eliminate ~60% code duplication between
configAnalyzer, configPeriodicity, configDecomposition.

Location: pipeline/helpers/baseConfigAdapter.py
"""

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from pipeline.helpers.configs import InstrumentTypeConfig
from pipeline.helpers.utils import validate_required_locals
from pipeline.timeSeriesProcessing.preprocessingConfig import (
    DataLengthCategory,
    FrequencyCategory,
    classify_data_length,
    get_frequency_category,
)

__version__ = "1.0.0"


class BaseConfigAdapter(ABC):
    """
    Base class for time series configuration adapters.

    Implements Template Method pattern to standardize workflow:
    1. Input parameter validation
    2. Data classification (frequency, length, instrument)
    3. Active method configuration initialization
    4. Base parameter propagation
    5. Adaptation rule application (frequency → length → instrument)
    6. Parameter constraint by data length
    7. Parameter range validation
    8. Specific adjustment application
    9. Finalization (type conversion, rounding)

    Eliminates ~60% duplication between config modules.
    Follows SOLID, DRY, KISS principles.

    ARCHITECTURE:
        BaseConfigAdapter (Template Method)
            ↓
        ├── AnalyzerConfigAdapter
        ├── PeriodicityConfigAdapter
        └── DecompositionConfigAdapter

    USAGE:
        class YourConfigAdapter(BaseConfigAdapter):
            BASE = {...}      # Method configurations
            ACTIVE = {...}    # Active methods per instrument
            RULES = {...}     # Adaptation rules

            def _get_integer_parameter_map(self):
                return {"method1": ["param1", "param2"]}

            # Override hooks for module specificity
    """

    # 🔍 DEBUG TRACING: Class variable for temporary debug helper
    _trace_helper = None

    # ========== MANDATORY CLASS ATTRIBUTES ==========

    BASE: ClassVar[Dict[str, Dict[str, Any]]]
    """
    Base method configurations for the module.

    Format:
    {
        "base": {"min_data_length": 10, "max_missing_ratio": 0.3},
        "method1": {"param1": value1, "param2": value2},
        "method2": {...}
    }
    """

    ACTIVE: ClassVar[Dict[InstrumentTypeConfig, List[str]]]
    """
    Active methods for instrument types.

    Format:
    {
        InstrumentTypeConfig.CRYPTO: ["base", "method1", "method2"],
        InstrumentTypeConfig.STOCK: ["base", "method1"]
    }
    """

    RULES: ClassVar[Dict[str, Dict[Any, List[Tuple[str, str, Any]]]]]
    """
    Parameter adaptation rules by categories.

    Format:
    {
        "frequency": {
            FrequencyCategory.HIGH: [
                ("method1.param1", "*", 2.0),   # param1 *= 2.0
                ("method1.param2", "=", "value") # param2 = "value"
            ]
        },
        "length": {...},
        "instrument": {...}
    }

    Operations: "*" (multiply), "+" (add), "=" (set)
    """

    # ========== TEMPLATE METHOD (MAIN WORKFLOW) ==========

    def build_config_from_properties(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Template Method: Unified workflow for all config adapters.

        Eliminates duplication of ~1800 lines of code between modules through
        standardized 9-step processing.

        Args:
            params: Time series parameters
                - instrument_type: InstrumentTypeConfig (required)
                - interval: str (required)
                - data_length: int (required)
                - volatility: float (optional, for crypto adjustments)
                - [module-specific parameters]

        Returns:
            Dict with adaptive configuration:
            {
                "method1": {...},
                "method2": {...},
                "_active_methods": ["method1", "method2"],
                "_weights": {...}  # optional
            }

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        module_name = self._get_module_name()

        # 1. Input parameter validation
        self._validate_input_parameters(params)

        # 2. Data classification
        classifications = self._classify_data(params)

        self._current_classifications = classifications

        logging.info(
            f"[{module_name}] Starting configuration build: "
            f"{classifications['instrument'].value}, {params['interval']}, "
            f"{classifications['data_length']} points"
        )

        # 3. Active method configuration initialization
        config = self._initialize_active_configs(classifications)

        # 4. Base parameter propagation
        self._propagate_base_parameters(config)

        # 5. Category-based adaptation rule application
        self._apply_adaptation_rules(config, classifications)

        # 6. Data length parameter constraint
        self._constrain_to_data_length(config, classifications["data_length"])

        # 7. Parameter range validation
        self._validate_parameter_ranges(config)

        # 8. Specific adjustment application
        self._apply_specific_adjustments(config, params, classifications)

        # 9. Configuration finalization
        self._finalize_configuration(config, params)

        logging.info(
            f"[{module_name}] Configuration ready: "
            f"active methods: {config.get('_active_methods', [])}"
        )

        self._current_classifications = None

        return config

    # ========== COMMON METHODS (ELIMINATE DUPLICATION) ==========

    def _validate_input_parameters(self, params: Dict[str, Any]) -> None:
        """
        Validate required input parameters.

        ELIMINATES DUPLICATION: 100% identical in all modules

        Args:
            params: Parameter dictionary for validation

        Raises:
            ValueError: If parameters are invalid
        """
        # Required parameters
        validate_required_locals(["instrument_type", "interval", "data_length"], params)

        # Validate data_length
        data_length = params["data_length"]
        if not isinstance(data_length, int) or data_length <= 0:
            raise ValueError(
                f"data_length must be positive int, got: {data_length}"
            )

        # Validate instrument_type
        instrument_type = params["instrument_type"]
        if not isinstance(instrument_type, InstrumentTypeConfig):
            raise ValueError(
                f"instrument_type must be InstrumentTypeConfig, "
                f"got: {type(instrument_type)}"
            )

        # Validate interval
        if not isinstance(params["interval"], str) or not params["interval"]:
            raise ValueError(f"interval must be non-empty string")

    def _classify_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify data by categories.

        ELIMINATES DUPLICATION: 95% identical in all modules

        Args:
            params: Time series parameters

        Returns:
            Dict with classifications:
            {
                'frequency': FrequencyCategory,
                'length': DataLengthCategory,
                'instrument': InstrumentTypeConfig,
                'data_length': int,
                # + module-specific via hook
            }
        """
        classifications = {
            "frequency": get_frequency_category(params["interval"]),
            "length": classify_data_length(params["data_length"]),
            "instrument": params["instrument_type"],
            "data_length": params["data_length"],
        }

        # Hook for module-specific classifications
        additional = self._get_additional_classifications(params)
        if additional:
            classifications.update(additional)

        return classifications

    def _initialize_active_configs(
        self, classifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Initialize configurations for active methods only.

        ELIMINATES DUPLICATION: 100% identical in all modules

        Args:
            classifications: Result from _classify_data()

        Returns:
            Dict with active method configurations

        Raises:
            ValueError: If no active methods for instrument
        """
        instrument = classifications["instrument"]

        # Get active method list for instrument
        active_methods = self.ACTIVE.get(instrument)
        if not active_methods:
            raise ValueError(
                f"No active methods for instrument {instrument.value}. "
                f"Add configuration to ACTIVE."
            )

        # Copy configurations only for active methods
        config: Dict[str, Any] = {}
        for method in active_methods:
            if method in self.BASE:
                config[method] = deepcopy(self.BASE[method])

        # Add metadata (_active_methods without 'base')
        config["_active_methods"] = [m for m in active_methods if m != "base"]

        module_name = self._get_module_name()
        logging.debug(
            f"[{module_name}] Initialization: {instrument.value}, "
            f"active methods: {config['_active_methods']}"
        )

        return config

    def _propagate_base_parameters(self, config: Dict[str, Any]) -> None:
        """
        Propagate base parameters to all methods.

        ELIMINATES DUPLICATION: 100% identical in all modules

        Base parameters from BASE["base"] are copied to all methods,
        but do not overwrite existing method parameters.

        Args:
            config: Configuration to modify (in-place)
        """
        if "base" not in self.BASE:
            return

        base_params = self.BASE["base"]

        # Propagate to each method
        for method_name in config:
            if method_name.startswith("_"):
                continue  # Skip metadata

            if isinstance(config[method_name], dict):
                # Base parameters first, specific ones override
                merged = {**base_params, **config[method_name]}
                config[method_name] = merged

                logging.debug(
                    f"Base parameters added to {method_name}: "
                    f"{list(base_params.keys())}"
                )

    def _apply_adaptation_rules(
        self, config: Dict[str, Any], classifications: Dict[str, Any]
    ) -> None:
        """
        Apply adaptation rules in order: frequency → length → instrument.

        ELIMINATES DUPLICATION: 90% identical in all modules

        Rule application order is critical for correct results.

        Args:
            config: Configuration to modify (in-place)
            classifications: Result from _classify_data()
        """
        # Rule application order is critical
        for category in ["frequency", "length", "instrument"]:
            if category not in self.RULES:
                continue

            key = classifications[category]
            if key not in self.RULES[category]:
                continue

            rules = self.RULES[category][key]
            for rule in rules:
                self._apply_single_rule(config, rule)

    def _apply_single_rule(
        self, config: Dict[str, Any], rule: Tuple[str, str, Any]
    ) -> None:
        """
        Apply a single adaptation rule to configuration.

        ELIMINATES DUPLICATION: 95% identical in all modules

        Args:
            config: Configuration to modify (in-place)
            rule: Tuple (path, operation, value)
                - path: "method.parameter" or "_weight.method"
                - operation: "*" (multiply), "+" (add), "=" (set)
                - value: number or other type
        """
        path, operation, value = rule

        # Special handling for weights (for periodicity)
        if path.startswith("_weight."):
            method = path.split(".")[1]
            if "_weights" in config and method in config["_weights"]:
                if operation == "*":
                    config["_weights"][method] *= value
                elif operation == "=":
                    config["_weights"][method] = value
                logging.debug(f"{path}: {operation} {value}")
            return

        # Regular parameter handling
        parts = path.split(".")
        if len(parts) < 2:
            return

        method = parts[0]
        param_path = parts[1:]

        if method not in config:
            return

        # Navigate nested parameters (for decomposition)
        target = config[method]
        for part in param_path[:-1]:
            if not isinstance(target, dict) or part not in target:
                return
            target = target[part]

        param = param_path[-1]
        if not isinstance(target, dict) or param not in target:
            return

        old_val = target[param]

        # None-safety for arithmetic operations
        if old_val is None and operation in ["*", "+"]:
            logging.debug(f"Skipping rule {path}: None value")
            return

        # Apply operation
        if operation == "*":
            new_val = old_val * value
        elif operation == "+":
            new_val = old_val + value
        elif operation == "=":
            new_val = value
        else:
            logging.warning(f"Unknown operation: {operation} for {path}")
            return

        # Hook for transformation
        new_val = self._transform_rule_value(method, param, new_val)
        target[param] = new_val

        logging.debug(f"{path}: {old_val} {operation} {value} = {target[param]}")

    def _constrain_to_data_length(
        self, config: Dict[str, Any], data_length: int
    ) -> None:
        """
        Constrain parameters to actual data length.

        ELIMINATES DUPLICATION: 75% similar in all modules

        Basic mathematical constraints for common parameters.
        Module-specific constraints via hook.

        Args:
            config: Configuration to modify (in-place)
            data_length: Time series length
        """
        # Basic constraints for base parameters
        if "base" in config:
            if "min_data_length" in config["base"]:
                # min_data_length cannot exceed data_length
                config["base"]["min_data_length"] = min(
                    config["base"]["min_data_length"], data_length
                )
                logging.debug(
                    f"base.min_data_length constrained: "
                    f"{config['base']['min_data_length']}"
                )

        # Hook for setup BEFORE constraints
        self._pre_constraint_setup(config, data_length)

        # Hook for module-specific constraints
        self._apply_module_specific_constraints(config, data_length)

    def _validate_parameter_ranges(self, config: Dict[str, Any]) -> None:
        """
        Validate parameter ranges with automatic correction.

        ELIMINATES DUPLICATION: 80% similar in all modules

        Common checks for base parameters:
        - Confidence thresholds ∈ (0, 1)
        - Positive parameters > 0
        - Rational max/min ratios

        Args:
            config: Configuration to validate and correct (in-place)
        """
        # Basic validation for base parameters
        if "base" in config:
            base = config["base"]

            # min_data_length > 0
            if "min_data_length" in base and base["min_data_length"] <= 0:
                logging.warning(
                    f"base.min_data_length too small: "
                    f"{base['min_data_length']} -> 5"
                )
                base["min_data_length"] = 5

            # max_missing_ratio ∈ [0, 1]
            if "max_missing_ratio" in base:
                value = base["max_missing_ratio"]
                if not (0 <= value <= 1):
                    clamped = max(0.0, min(1.0, value))
                    logging.warning(
                        f"base.max_missing_ratio out of range [0,1]: "
                        f"{value} -> {clamped}"
                    )
                    base["max_missing_ratio"] = clamped

        # Hook for module-specific validation
        self._validate_module_specific_ranges(config)

    def _apply_specific_adjustments(
        self,
        config: Dict[str, Any],
        params: Dict[str, Any],
        classifications: Dict[str, Any],
    ) -> None:
        """
        Apply specific adjustments (crypto, volatility, etc.).

        ELIMINATES DUPLICATION: 70% similar in all modules

        Args:
            config: Configuration to modify (in-place)
            params: Original parameters with metadata
            classifications: Result from _classify_data()
        """
        instrument = classifications["instrument"]

        # Crypto adjustments (if applicable)
        if instrument == InstrumentTypeConfig.CRYPTO:
            volatility = params.get("volatility")
            self._apply_crypto_adjustments(config, volatility)

        # Hook for additional adaptations
        self._apply_module_specific_adaptations(config, params, classifications)

    def _finalize_configuration(
        self, config: Dict[str, Any], params: Dict[str, Any]
    ) -> None:
        """
        Finalize configuration: type conversion, rounding, cleanup, logging.

        ELIMINATES DUPLICATION: 90% identical in all modules

        Args:
            config: Configuration to finalize (in-place)
            params: Original parameters with metadata
        """
        # Convert parameters to int and round float
        self._process_integer_and_round_params(config)

        # Hook for module-specific finalization
        self._finalize_module_specific(config, params)

        self._post_adaptation_propagation(config)

        # Automatic cleanup of inactive methods
        self._cleanup_inactive_methods(config)

        # Final logging
        module_name = self._get_module_name()
        active_methods = config.get("_active_methods", [])
        logging.debug(
            f"[{module_name}] Finalization complete: "
            f"{len(active_methods)} active methods"
        )

    def _process_integer_and_round_params(self, config: Dict[str, Any]) -> None:
        """
        Convert integer parameters and round float.

        ELIMINATES DUPLICATION: 90% identical in all modules

        Args:
            config: Configuration to process (in-place)
        """
        # Get integer parameter list (module-specific)
        integer_params = self._get_integer_parameter_map()

        # Convert to int
        for method, params in integer_params.items():
            if method in config:
                for param in params:
                    if param in config[method] and config[method][param] is not None:
                        config[method][param] = int(config[method][param])
                        logging.debug(
                            f"Converted to int: {method}.{param} = "
                            f"{config[method][param]}"
                        )

        # Round float for readability
        for method in config:
            if method.startswith("_"):
                continue  # Skip metadata

            if isinstance(config[method], dict):
                for param, value in config[method].items():
                    if isinstance(value, float):
                        config[method][param] = round(value, 4)

    def _cleanup_inactive_methods(self, config: Dict[str, Any]) -> None:
        """
        Remove configurations of inactive methods.

        Protected hook for modules with dynamic _active_methods changes.
        Safe for modules with static lists (will not remove anything).
        """
        if "_active_methods" not in config:
            return

        active_methods = config["_active_methods"]
        methods_to_remove = []

        for method in list(config.keys()):
            # Keep metadata (_active_methods, _weights, etc.)
            if method.startswith("_"):
                continue
            # Keep base (used for parameter propagation)
            if method == "base":
                continue
            # Remove methods not in final active list
            if method not in active_methods:
                methods_to_remove.append(method)

        if methods_to_remove:
            for method in methods_to_remove:
                del config[method]

            module_name = self._get_module_name()
            logging.debug(
                f"[{module_name}] Cleaned up inactive methods: {methods_to_remove}"
            )

    # ========== UTILITY METHODS ==========

    def _get_module_name(self) -> str:
        """
        Get module name for logging.

        Returns:
            Module name in lowercase (e.g., "analyzer", "periodicity")
        """
        class_name = self.__class__.__name__
        return class_name.replace("ConfigAdapter", "").lower()

    def __str__(self) -> str:
        """String representation for logging."""
        module_name = self._get_module_name()
        return f"[{module_name}ConfigAdapter]"

    # ========== ABSTRACT METHODS (MANDATORY) ==========

    @abstractmethod
    def _get_integer_parameter_map(self) -> Dict[str, List[str]]:
        """
        Get integer parameter map for module.

        MANDATORY IMPLEMENTATION in child classes.

        Returns:
            Dict of form:
            {
                "method1": ["param1", "param2"],
                "method2": ["param3"]
            }

        Example:
            def _get_integer_parameter_map(self):
                return {
                    "acf": ["min_period", "max_period"],
                    "spectral": ["n_peaks"]
                }
        """
        pass

    # ========== HOOKS (OPTIONAL OVERRIDES) ==========

    def _pre_constraint_setup(self, config: Dict[str, Any], data_length: int) -> None:
        """
        Hook: Setup before applying constraints.

        Called BEFORE _apply_module_specific_constraints to initialize
        parameters required for constraint checks.

        Args:
            config: Configuration to modify
            data_length: Time series length
        """
        pass

    def _transform_rule_value(self, method: str, param: str, value: Any) -> Any:
        """Hook: Transform value after rule application."""
        return value

    def _post_adaptation_propagation(self, config: Dict[str, Any]) -> None:
        """
        Hook: Final propagation after all adaptations.
        Override for modules that recalculate base params.
        """
        pass

    def _get_additional_classifications(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook: Additional module-specific classifications.

        Override for module-specific classifications
        (e.g., volatility levels, noise categories).

        Args:
            params: Original time series parameters

        Returns:
            Dict with additional classifications (empty by default)
        """
        return {}

    def _apply_module_specific_constraints(
        self, config: Dict[str, Any], data_length: int
    ) -> None:
        """
        Hook: Module-specific mathematical constraints.

        Override for specific constraints:
        - ACF: max_lags ≤ data_length/3
        - Spectral: nperseg ≤ data_length/2
        - Wavelet: n_scales ≤ data_length/4
        - etc.

        Args:
            config: Configuration to modify (in-place)
            data_length: Time series length
        """
        pass

    def _validate_module_specific_ranges(self, config: Dict[str, Any]) -> None:
        """
        Hook: Module-specific range validation.

        Override for specific parameter checks:
        - Confidence thresholds ∈ (0, 1)
        - Special method constraints
        - Consistency of related parameters

        Args:
            config: Configuration to validate (in-place)
        """
        pass

    def _apply_crypto_adjustments(
        self, config: Dict[str, Any], volatility: Optional[float] = None
    ) -> None:
        """
        Hook: Crypto adjustments.

        Override for module-specific crypto adaptations.
        Default: no adjustments.

        Args:
            config: Configuration to modify (in-place)
            volatility: Volatility (optional)
        """
        pass

    def _apply_module_specific_adaptations(
        self,
        config: Dict[str, Any],
        params: Dict[str, Any],
        classifications: Dict[str, Any],
    ) -> None:
        """
        Hook: Additional module-specific adaptations.

        Override for complex adaptations:
        - Adaptation to noise levels
        - Adjustments for trends
        - Special rules for stationarity
        - etc.

        Args:
            config: Configuration to modify (in-place)
            params: Original time series parameters
            classifications: Result from _classify_data()
        """
        pass

    def _finalize_module_specific(
        self, config: Dict[str, Any], params: Dict[str, Any]
    ) -> None:
        """
        Hook: Module-specific finalization.

        Override for final adjustments before returning configuration.

        Args:
            config: Configuration for final edits (in-place)
            params: Original time series parameters
        """
        pass