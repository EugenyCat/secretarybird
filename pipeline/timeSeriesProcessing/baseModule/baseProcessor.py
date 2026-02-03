"""
BaseProcessor - unified base class for all time series processors.

Implements Template Method pattern to eliminate 95% code duplication between
analyzer, periodicity and decomposition processors.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

from pipeline.helpers.configs import PropertySourceConfig
from pipeline.helpers.utils import validate_required_locals

__version__ = "1.0.0"


class BaseProcessor(ABC):
    """
    Base class for all time series processors.

    Implements Template Method pattern to standardize workflow:
    1. Input parameter validation
    2. Properties check for repeated run
    3. Module algorithm execution (abstract method)
    4. DataFrame enrichment with additional columns
    5. Context update with results

    Eliminates 95% duplication between processors.
    Follows SOLID, DRY, KISS principles.
    """

    # 🔍 DEBUG TRACING: Class variable for temporary debug helper
    _trace_helper = None

    def __init__(
        self,
        ts_id: str,
        currency: str,
        interval: str,
        instrument_type,
        targetColumn: str,
        properties: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        fallbackBehavior: str = "error",
        module_name: Optional[str] = None,
    ) -> None:
        """
        Universal processor initialization.

        Args:
            ts_id: Time series identifier
            currency: Instrument currency
            interval: Data interval
            instrument_type: Instrument type
            targetColumn: Target column for processing
            properties: Existing properties from DB
            config: Module configuration
            fallbackBehavior: Error behavior ('error' or 'simple')
            module_name: Module name for context (auto-determined if not specified)
        """
        # Common validation (eliminates duplication)
        validate_required_locals(
            ["ts_id", "currency", "interval", "instrument_type", "targetColumn"],
            locals(),
        )

        # Common fields (eliminates duplication of 8 fields × 3 processors)
        self.ts_id = ts_id
        self.currency = currency
        self.interval = interval
        self.instrument_type = instrument_type
        self.targetColumn = targetColumn
        self.properties = properties if self._validate_properties(properties) else None
        self.config = config
        self.fallbackBehavior = fallbackBehavior
        self.module_name = module_name or self._get_module_name()
        self.algorithm = None  # Lazy initialization

        # Common logging (eliminates duplication)
        logging.info(
            f"{self.__str__()} Initialized for column '{targetColumn}', "
            f"fallback='{fallbackBehavior}'"
        )

    def _get_algorithm_input(
        self, data: pd.DataFrame
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Template Method Extension: Get algorithm input from DataFrame.

        DEFAULT BEHAVIOR: Returns target column as Series (preserves backward compatibility)
        OVERRIDE: Custom processors can return full DataFrame or custom structure

        This allows processors like OutlierDetection to access full DataFrame
        with enrichment columns while maintaining architecture consistency.

        Returns:
            pd.Series: For most processors (default)
            pd.DataFrame: For processors needing enrichment columns (override)
        """
        return data[self.targetColumn]

    def process(
        self, data: pd.DataFrame, context: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Template Method: Unified workflow for all processors.

        Eliminates duplication of ~150 lines of code in each processor.
        """
        # 1. Common validation (eliminates duplication)
        validate_required_locals(["data", "context"], locals())

        logging.info(
            f"{self.__str__()} Starting processing: ts_id={self.ts_id}, "
            f"interval={self.interval}, instrument={self.instrument_type.value}"
        )

        # 2. Repeated run (eliminates duplication of ~25 lines)
        if self.properties:
            return self._handle_repeated_run(data, context)

        try:
            # 3. Check target column presence
            if self.targetColumn not in data.columns:
                error_msg = f"Target column '{self.targetColumn}' not found"
                return self._handle_processor_error(error_msg, data, context)

            # 4. Algorithm execution (strategy - overridden in child classes)
            algorithm_input = self._get_algorithm_input(data)
            algorithm_result = self._execute_algorithm(algorithm_input, context)

            # 5. Error handling (eliminates duplication of ~15 lines)
            if algorithm_result["status"] == "error":
                return self._handle_processor_error(
                    algorithm_result.get("message", "Unknown error"),
                    data,
                    context,
                    algorithm_result.get("metadata", {}),
                )

            # 6. Property extraction (strategy - overridden in child classes)
            module_properties = self._extract_properties(algorithm_result)

            # 7. DataFrame enrichment (eliminates duplication of ~35 lines)
            enriched_data = self._add_module_columns_to_dataframe(
                data, algorithm_result
            )

            # 8. Context update (eliminates duplication of ~10 lines)
            context["propertySources"][
                self.module_name
            ] = PropertySourceConfig.CALCULATED
            context["currentProperties"][self.module_name] = module_properties

            # 9. Common logging (eliminates duplication)
            self._log_success_summary(module_properties)

            return enriched_data, context

        except Exception as e:
            error_msg = f"Critical error in {self.module_name} processor: {str(e)}"
            logging.error(f"{self.__str__()} {error_msg}", exc_info=True)
            return self._handle_processor_error(
                error_msg, data, context, {"error_type": type(e).__name__}
            )

    # ========== ABSTRACT METHODS (STRATEGIES) ==========

    @abstractmethod
    def _execute_algorithm(
        self, series: pd.Series, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute module algorithm.

        Returns:
            Dict in standard format: {"status": "success/error", "result": {...}, "metadata": {...}}
        """
        pass

    @abstractmethod
    def _extract_properties(self, algorithm_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract module properties for DB storage.

        Returns:
            Dict with properties for context["currentProperties"][module_name]
        """
        pass

    @abstractmethod
    def _validate_properties(self, properties: Optional[Dict[str, Any]]) -> bool:
        """Validate properties received from DB."""
        pass

    @abstractmethod
    def _get_default_properties(self) -> Dict[str, Any]:
        """Default properties for fallbackBehavior='simple'."""
        pass

    @abstractmethod
    def _log_success_summary(self, properties: Dict[str, Any]) -> None:
        """Module-specific logging of successful completion."""
        pass

    @abstractmethod
    def _initialize_algorithm(self):
        """Initialize module algorithm."""
        pass

    def _get_heuristic_fallback_values(self) -> Dict[str, Any]:
        """
        Get heuristic fallback values for module.

        Abstract method for module-specific generation of heuristic fallback values
        during processing errors. Allows each processor to define
        its own fallback logic considering module specifics.

        DEFAULT BEHAVIOR: Returns empty dict (no heuristics).
        OVERRIDE in child classes when needed:
        - PeriodicityDetectorProcessor: adds suggested_periods and main_period
        - AnalyzerProcessor: may add basic statistics (optional)
        - DecompositionProcessor: may add simple decomposition (optional)

        Returns:
            Dict with heuristic fallback values to enrich default_properties
        """
        # Default implementation: no heuristic fallback
        logging.debug(f"{self.__str__()} No heuristic fallback values available")
        return {}

    # ========== COMMON METHODS (ELIMINATE DUPLICATION) ==========

    def _handle_repeated_run(
        self, data: pd.DataFrame, context: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Template Method: Unified repeated run logic.

        Eliminates 95% duplication between processors through Template Method pattern.
        Specific restoration logic isolated in abstract methods.

        ARCHITECTURAL UPDATE: Added enrichment column restoration stage
        to ensure consistency between all processors.
        """
        # 1. Restore module-specific state (overridden in child classes)
        self._restore_module_state()

        # 2. Common context operations (unified logic)
        context["propertySources"][self.module_name] = PropertySourceConfig.DATABASE
        context["currentProperties"][self.module_name] = self.properties

        # 3. Restore enrichment columns (optional, overridden when needed)
        try:
            data = self._restore_enrichment_columns(data, context)
            logging.debug(f"{self.__str__()} Enrichment columns restoration completed")
        except Exception as e:
            logging.warning(
                f"{self.__str__()} Enrichment restoration failed: {str(e)}, continuing without enrichment"
            )

        # 4. Standard logging of successful repeated run
        logging.info(
            f"{self.__str__()} Repeated run: reused properties with enrichment restoration"
        )
        return data, context

    @abstractmethod
    def _restore_module_state(self) -> None:
        """
        Restore module-specific state on repeated run.

        Each processor implements its own logic:
        - AnalyzerProcessor: restore config_analyzer
        - PeriodicityProcessor: restore config_periodicity
        - DecompositionProcessor: restore components + config_decomposition
        """
        pass

    def _restore_enrichment_columns(
        self, data: pd.DataFrame, context: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Restore enrichment columns on repeated run.

        Template Method Extension: optional method to ensure consistency
        in DataFrame enrichment restoration between all processors.

        DEFAULT BEHAVIOR: No restoration (for processors without enrichment).
        OVERRIDE in child classes when needed:
        - AnalyzerProcessor: restore outlier boolean columns
        - DecompositionProcessor: restore component time series

        Args:
            data: DataFrame to restore enrichment columns
            context: Processing context with properties

        Returns:
            DataFrame with restored enrichment columns (or original DataFrame)
        """
        # Default implementation: no enrichment restoration
        logging.debug(f"{self.__str__()} No enrichment restoration required")
        return data

    def _handle_processor_error(
        self,
        error_msg: str,
        data: pd.DataFrame,
        context: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Universal error handling (eliminates duplication of ~45 lines)."""
        logging.error(f"{self.__str__()} {error_msg}")

        # fallbackBehavior == 'simple': use default properties
        if self.fallbackBehavior == "simple":
            logging.warning(
                f"{self.__str__()} Using default properties due to error"
            )
            try:
                default_props = self._get_default_properties()

                # Add heuristic fallback values
                heuristic_values = self._get_heuristic_fallback_values()
                if heuristic_values:
                    default_props.update(heuristic_values)
                    logging.info(
                        f"{self.__str__()} Added heuristic fallback values: "
                        f"{list(heuristic_values.keys())}"
                    )

                context["propertySources"][
                    self.module_name
                ] = PropertySourceConfig.DEFAULT
                context["currentProperties"][self.module_name] = default_props
            except Exception as fallback_error:
                logging.error(f"{self.__str__()} Error in fallback: {fallback_error}")
                context["propertySources"][
                    self.module_name
                ] = PropertySourceConfig.DEFAULT
                context["currentProperties"][self.module_name] = {}

        # Add heuristic values even with fallbackBehavior='error'
        # to enrich error context (according to updated ticket)
        elif self.fallbackBehavior == "error":
            try:
                heuristic_values = self._get_heuristic_fallback_values()
                if heuristic_values:
                    # Add heuristic values to error metadata for diagnostics
                    if not metadata:
                        metadata = {}
                    metadata["heuristic_fallback"] = heuristic_values
                    logging.info(
                        f"{self.__str__()} Added heuristic values to error context: "
                        f"{list(heuristic_values.keys())}"
                    )
            except Exception as heuristic_error:
                logging.warning(
                    f"{self.__str__()} Error getting heuristic values: {heuristic_error}"
                )

        # Add error information to context
        context["error"] = {
            "stage": self.module_name,
            "message": error_msg,
            "metadata": metadata or {},
            "critical": self.fallbackBehavior != "simple",
        }

        return data, context

    def _add_module_columns_to_dataframe(
        self, dataframe: pd.DataFrame, algorithm_result: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Common DataFrame enrichment pattern (eliminates duplication of ~35 lines).

        Child classes can override for specific logic.
        """
        try:
            if algorithm_result["status"] != "success":
                logging.warning(f"{self} - Algorithm failed, skipping enrichment")
                return dataframe

            # Create algorithm and perform DataFrame enrichment
            if self.algorithm is None:
                self.algorithm = self._initialize_algorithm()

            # Check DataFrame enrichment support
            if self.algorithm is not None and hasattr(
                self.algorithm, "process_with_dataframe_enrichment"
            ):
                enriched_df, _ = self.algorithm.process_with_dataframe_enrichment(
                    dataframe, self.targetColumn, context=None
                )
                logging.debug(f"{self} - Added {self.module_name} columns to DataFrame")
                return enriched_df
            else:
                logging.debug(
                    f"{self} - Algorithm doesn't support DataFrame enrichment"
                )
                return dataframe

        except Exception as e:
            logging.warning(
                f"{self} - Enrichment failed ({str(e)}), returning original DataFrame"
            )
            return dataframe

    def _get_module_name(self) -> str:
        """Automatic module name determination from class."""
        class_name = self.__class__.__name__
        if "Analysis" in class_name:
            return "analyzer"
        elif "Periodicity" in class_name:
            return "periodicity"
        elif "Decomposition" in class_name:
            return "decomposition"
        else:
            return "unknown"

    def __str__(self) -> str:
        """Common string representation (eliminates duplication)."""
        return f"[{self.module_name.title()}Processor][{self.ts_id}|{self.interval}]"