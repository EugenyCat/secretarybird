"""
OutlierDetectionProcessor - Level 1 Processor with BaseProcessor Template Method.

REFACTORED v2.0.0 - Full architectural compliance (Oct 20, 2025)

Implements BaseProcessor pattern for outlier detection in preprocessing pipeline.
Provides:
- DataFrame enrichment with outlier columns
- Context integration (propertySources + currentProperties)
- Fail-fast validation and graceful degradation
- Production-ready error handling

Architecture:
    Level 1: OutlierDetectionProcessor (THIS) - BaseProcessor inheritance
        ↓
    Level 2: OutlierDetectionAlgorithm - Enhanced Decision Tree
        ↓
    Level 3: Detection Methods (Tier 1-2)

Version: 2.0.0 (Refactored - Oct 20, 2025)
Changes:
    - REMOVED: _validate_and_get_dataframe() method (HIGH-1 fix)
    - ADDED: _validate_context_dependencies() method (HIGH-5 fix)
    - REFACTORED: _execute_algorithm() to follow BaseProcessor contract (HIGH-2)
    - FIXED: Context access via context['currentProperties'] (HIGH-3)
    - ADDED: Component extraction logic from context (HIGH-4)
    - IMPROVED: _additional_validation() with actual validation (MEDIUM-1)
    - ARCHITECTURAL: Now follows DecompositionProcessor pattern exactly
"""

import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from pipeline.helpers.configs import InstrumentTypeConfig
from pipeline.helpers.utils import validate_required_locals
from pipeline.timeSeriesProcessing.baseModule.baseProcessor import BaseProcessor
from pipeline.timeSeriesProcessing.outlierDetection.algorithmOutlierDetection import (
    OutlierDetectionAlgorithm,
)
from pipeline.timeSeriesProcessing.outlierDetection.configOutlierDetection import (
    OutlierDetectionConfigAdapter,
)

__version__ = "2.0.0"


class OutlierDetectionProcessor(BaseProcessor):
    """
    Level 1 Processor: Outlier Detection with BaseProcessor Template Method.

    COMPLIANCE: Implements BaseProcessor pattern correctly
    STRATEGY: Enhanced Decision Tree with tiered execution

    Features:
    - 85-86% code reduction through BaseProcessor inheritance
    - 8-step standardized workflow
    - DataFrame enrichment (6 columns)
    - Context integration via context['currentProperties']
    - Fail-fast validation
    - Graceful degradation

    Enrichment Columns:
    - outliers: Boolean mask of consensus outliers
    - outlier_confidence: Float confidence scores
    - outlier_score_enhanced: Quality-weighted scores
    - outlier_type: Type classification (statistical/component/consensus/none)
    - price_robust: Interpolated robust price
    - microstructure_flag: HFT noise flags (placeholder)

    Example:
        >>> processor = OutlierDetectionProcessor(
        ...     ts_id="BTC-USD",
        ...     currency="BTC",
        ...     interval="1h",
        ...     instrument_type=InstrumentTypeConfig.CRYPTO,
        ...     targetColumn="close"
        ... )
        >>> enriched_data, updated_context = processor.process(data, context)
    """

    # Enrichment columns for DataFrame
    ENRICHMENT_COLUMNS = [
        "outliers",
        "outlier_confidence",
        "outlier_score_enhanced",
        "outlier_type",
        "price_robust",
    ]

    def __init__(
        self,
        ts_id: str,
        currency: str,
        interval: str,
        instrument_type: InstrumentTypeConfig,
        targetColumn: str,
        properties: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        fallbackBehavior: str = "error",
    ) -> None:
        """
        Initialize OutlierDetectionProcessor.

        COMPLIANCE FIX v2.0.0: Follows BaseProcessor initialization pattern

        Args:
            ts_id: Time series identifier
            currency: Currency of instrument
            interval: Data interval
            instrument_type: Instrument type (CRYPTO/STOCK)
            targetColumn: Target column for processing
            properties: Existing properties from DB
            config: Module configuration (optional, generated from adapter)
            fallbackBehavior: Error handling ('error' or 'simple')

        Raises:
            ValueError: If required parameters missing or invalid
        """
        # COMPLIANCE: Validate required parameters FIRST (fail-fast)
        validate_required_locals(
            ["ts_id", "currency", "interval", "instrument_type", "targetColumn"],
            locals(),
        )

        # Call parent __init__ with module_name
        super().__init__(
            ts_id=ts_id,
            currency=currency,
            interval=interval,
            instrument_type=instrument_type,
            targetColumn=targetColumn,
            properties=properties,
            config=config,
            fallbackBehavior=fallbackBehavior,
            module_name="outlier_detection",
        )

        # Module-specific initialization
        self.config_adapter = None
        self.algorithm = None  # Lazy initialization
        self._initialization_failed = False

        logging.info(f"{self} - Initialized for {ts_id}, interval={interval}")

    def __str__(self) -> str:
        """String representation for logging."""
        return f"[OutlierDetectionProcessor][{self.ts_id}][{self.interval}]"

    # ========== ABSTRACT METHODS BaseProcessor ==========

    def _validate_context_dependencies(self, context: Dict[str, Any]) -> None:
        """
        Validate required dependencies in context.

        Args:
            context: Processing context

        Raises:
            ValueError: If missing required context groups or data
        """
        required_groups = ["analyzer", "decomposition"]
        current_props = context["currentProperties"]

        missing_groups = []
        for group in required_groups:
            if group not in current_props or not current_props[group]:
                missing_groups.append(group)

        if missing_groups:
            raise ValueError(
                f"{self} - Missing required context groups: {missing_groups}. "
                f"Outlier detection requires analyzer and decomposition to run first."
            )

    def _get_algorithm_input(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Override: Return full DataFrame with boolean columns.

        OutlierDetection needs boolean columns (is_zscore_outlier, etc.)
        which are in DataFrame, not in context.
        """
        return data

    def _execute_algorithm(
        self, dataframe: pd.DataFrame, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute outlier detection algorithm."""
        try:
            # 🔍 TRACE: Input data before outlier detection
            if self._trace_helper:
                self._trace_helper.save_df(dataframe, "outlier_01_input_data")
                self._trace_helper.save_context(
                    {
                        "data_shape": dataframe.shape,
                        "columns": list(dataframe.columns),
                        "boolean_columns_present": [
                            col
                            for col in [
                                "is_zscore_outlier",
                                "is_iqr_outlier",
                                "is_mad_outlier",
                            ]
                            if col in dataframe.columns
                        ],
                        "decomposition_columns_present": [
                            col
                            for col in ["trend", "seasonal", "residual"]
                            if col in dataframe.columns
                        ],
                        "context_keys": (
                            list(context["currentProperties"].keys()) if context else []
                        ),
                    },
                    "outlier_01_input_validation",
                )

            # Step 1: Validate context dependencies
            self._validate_context_dependencies(context)

            # Step 2: Initialize algorithm
            if self.algorithm is None:
                self._initialize_algorithm(len(dataframe))

            # Step 3: Validate boolean columns in DataFrame
            required_booleans = [
                "is_zscore_outlier",
                "is_iqr_outlier",
                "is_mad_outlier",
            ]
            missing_bools = [
                col for col in required_booleans if col not in dataframe.columns
            ]

            if missing_bools:
                raise ValueError(
                    f"{self} - Missing boolean columns in DataFrame: {missing_bools}. "
                    f"AnalyzerProcessor must run first."
                )

            # Step 4: Validate decomposition component columns in DataFrame
            decomp_columns = [
                "trend",
                "seasonal",
                "residual",
            ]
            missing_decomp = [
                col for col in decomp_columns if col not in dataframe.columns
            ]

            if missing_decomp:
                raise ValueError(
                    f"{self} - Missing decomposition columns in DataFrame: {missing_decomp}. "
                    f"DecompositionProcessor must run first."
                )

            # Step 5: Construct DataFrame for algorithm - ALL DATA FROM DATAFRAME!
            algorithm_input = pd.DataFrame(
                {
                    # Target column from DataFrame
                    self.targetColumn: dataframe[self.targetColumn],
                    # Decomposition components from DataFrame (NOT from context!)
                    "trend": dataframe[f"trend"],
                    "seasonal": dataframe[f"seasonal"],
                    "residual": dataframe["residual"],
                    # Boolean columns from DataFrame
                    "is_zscore_outlier": dataframe["is_zscore_outlier"],
                    "is_iqr_outlier": dataframe["is_iqr_outlier"],
                    "is_mad_outlier": dataframe["is_mad_outlier"],
                }
            )

            logging.debug(
                f"{self} - Constructed DataFrame with {len(algorithm_input)} rows, "
                f"{len(algorithm_input.columns)} columns for algorithm"
            )

            # 🔍 TRACE: Algorithm input DataFrame
            if self._trace_helper:
                self._trace_helper.save_df(
                    algorithm_input, "outlier_02_algorithm_input"
                )

            # Step 6: Execute algorithm
            result = self.algorithm.process(algorithm_input, context)

            # 🔍 TRACE: Algorithm result
            if self._trace_helper:
                self._trace_helper.save_context(
                    {
                        "algorithm_status": result["status"],
                        "algorithm_metadata": result["metadata"],
                        "outlier_count": result["result"]["outlier_count"],
                        "tier_reached": result["metadata"]["tier_reached"],
                    },
                    "outlier_03_algorithm_result",
                )
                # Save enrichment columns if present
                if result["status"] == "success":
                    enrichment_df = pd.DataFrame()
                    for col in self.ENRICHMENT_COLUMNS:
                        if col in result["result"]:
                            enrichment_df[col] = result["result"][col]
                    if not enrichment_df.empty:
                        self._trace_helper.save_df(
                            enrichment_df, "outlier_04_enrichment_columns"
                        )

            # Step 7: Handle graceful degradation
            if result["status"] == "error" and self.fallbackBehavior == "simple":
                logging.warning(
                    f"{self} - Algorithm failed, using simple mode: "
                    f"{result['message']}"
                )
                return self._simple_mode_detection(algorithm_input)

            return result

        except Exception as e:
            logging.error(f"{self} - Algorithm execution failed: {e}", exc_info=True)

            if self.fallbackBehavior == "simple":
                logging.info(f"{self} - Falling back to simple mode")
                try:
                    simple_data = pd.DataFrame(
                        {self.targetColumn: dataframe[self.targetColumn]}
                    )
                    return self._simple_mode_detection(simple_data)
                except Exception as fallback_error:
                    logging.error(f"{self} - Simple mode also failed: {fallback_error}")
                    return {
                        "status": "error",
                        "message": f"All detection modes failed: {str(e)}",
                    }

            return {"status": "error", "message": str(e)}

    def _initialize_algorithm(self, data_length: int) -> None:
        """
        Initialize algorithm with lazy loading pattern.

        Args:
            data_length: Length of data for config adaptation

        Creates ConfigAdapter and Algorithm instances on first use.
        Sets _initialization_failed flag on error to prevent retry loops.
        """
        try:
            # Step 1: Initialize ConfigAdapter WITHOUT parameters
            if self.config_adapter is None:
                self.config_adapter = OutlierDetectionConfigAdapter()

            # Step 2: Generate adaptive config WITH real data_length
            adaptive_config = self.config_adapter.build_config_from_properties(
                params={
                    "instrument_type": self.instrument_type,
                    "interval": self.interval,
                    "data_length": data_length,
                    "currency": self.currency,
                    "ts_id": self.ts_id,
                }
            )

            # Step 3: Merge with user config if provided
            if self.config:
                adaptive_config.update(self.config)

            # Step 4: Store config for future use and DB persistence
            self.config = adaptive_config

            # Step 5: Initialize algorithm
            self.algorithm = OutlierDetectionAlgorithm(adaptive_config)

            logging.info(
                f"{self} - Algorithm initialized successfully with adaptive config"
            )

        except Exception as e:
            self._initialization_failed = True
            logging.error(
                f"{self} - Algorithm initialization failed: {e}", exc_info=True
            )
            raise

    def _additional_validation(
        self, data: pd.DataFrame, context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Additional validation checks for outlier detection.

        IMPROVEMENT v2.0.0 (MEDIUM-1): Added actual validation logic instead
        of empty implementation.

        Validates that required boolean columns and decomposition components
        are available in context['currentProperties'].

        Args:
            data: DataFrame being processed
            context: Processing context

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            # Check for required analyzer boolean columns
            required_bools = [
                "is_zscore_outlier",
                "is_iqr_outlier",
                "is_mad_outlier",
            ]
            analyzer_props = context["currentProperties"]["analyzer"]
            outlier_detection = analyzer_props["outlier_detection"]

            missing_bools = [
                col for col in required_bools if col not in outlier_detection
            ]
            if missing_bools:
                return (
                    False,
                    f"Missing analyzer boolean columns: {missing_bools}",
                )

            # Check for decomposition components
            decomp_props = context["currentProperties"]["decomposition"]
            components = decomp_props["components"]
            required_components = ["trend", "seasonal", "residual"]

            missing_components = [c for c in required_components if c not in components]
            if missing_components:
                return (
                    False,
                    f"Missing decomposition components: {missing_components}",
                )

            return True, ""

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _restore_module_state(self) -> None:
        """
        Restore config and algorithm state for repeated runs.

        Follows AnalyzerProcessor pattern:
        1. Restore config from properties["config_outlier_detection"]
        2. Initialize algorithm with restored config
        3. No need to call build_config_from_properties() again
        """
        try:
            # Restore config from properties
            if self.properties and "config_outlier_detection" in self.properties:
                self.config = self.properties["config_outlier_detection"]

                # Initialize algorithm with restored config (no ConfigAdapter needed)
                self.algorithm = OutlierDetectionAlgorithm(self.config)

                logging.info(
                    f"{self} - Module state restored: config + algorithm initialized"
                )
            else:
                logging.warning(
                    f"{self} - config_outlier_detection not found in properties, "
                    "will create adaptive config on first run"
                )
                # Don't initialize here - let _execute_algorithm() handle it

        except Exception as e:
            logging.warning(
                f"{self} - State restoration failed: {e}, "
                "algorithm will be initialized lazily"
            )
            self.algorithm = None

    def _validate_properties(self, properties: Optional[Dict[str, Any]]) -> bool:
        """
        Validate properties retrieved from database.

        Args:
            properties: Properties to validate

        Returns:
            True if valid, False otherwise
        """
        if properties is None:
            return False

        # Check required keys
        required_keys = ["outliers_detected", "outlier_ratio", "quality_score"]
        return all(key in properties for key in required_keys)

    def _get_default_properties(self) -> Dict[str, Any]:
        """
        Get default properties for fallback behavior.

        NEW v2.0 (SB8-104): Includes adaptive weighting defaults.

        Returns:
            Dict with safe default properties
        """
        return {
            "outliers_detected": 0,
            "outlier_ratio": 0.0,
            "quality_score": 0.0,
            "context_reuse_ratio": 0.0,
            "execution_time_ms": 0.0,
            "tier_reached": 0,
            "methods_used": [],
            "regime": "unknown",
            "microstructure_analysis": {
                "hft_noise_level": 0.0,
                "hft_patterns": [],
                "applicable_for_data": False,
            },
            # NEW v2.0: Adaptive weighting defaults
            "adaptive_weights": {
                "statistical_enhancement": 1.0,
                "component_anomaly": 0.0,
            },
            "decomposition_quality_score": 0.0,
            "residual_strength": 0.0,
            "quality_warnings": [
                "CRITICAL: Fallback behavior, no adaptive weighting applied"
            ],
            "skipped_methods": [],
            "fallback_used": True,
        }

    def _log_success_summary(self, properties: Dict[str, Any]) -> None:
        """
        Log success summary with module-specific metrics.

        Args:
            properties: Extracted properties dict
        """
        outliers_detected = properties["outliers_detected"]
        outlier_ratio = properties["outlier_ratio"]
        quality_score = properties["quality_score"]
        tier_reached = properties["tier_reached"]
        methods_used = properties["methods_used"]
        regime = properties["regime"]

        logging.info(
            f"{self} - SUCCESS: Detected {outliers_detected} outliers "
            f"({outlier_ratio:.1%}), quality={quality_score:.2f}, "
            f"tier={tier_reached}, methods={methods_used}, regime={regime}"
        )

    def _extract_properties(self, algorithm_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract properties from algorithm result for database storage.

        NEW v2.0 (SB8-104): Extracts adaptive weighting metadata for context propagation.

        Args:
            algorithm_result: Result dict from algorithm

        Returns:
            Dict with properties for context["currentProperties"][module_name]
        """
        if algorithm_result["status"] != "success":
            return self._get_default_properties()

        result = algorithm_result["result"]
        metadata = algorithm_result["metadata"]

        # FIXED: Correct data extraction from result
        data_length = (
            len(result["outliers"]) if isinstance(result["outliers"], pd.Series) else 0
        )
        outlier_count = int(result["outlier_count"])
        outlier_ratio = outlier_count / data_length if data_length > 0 else 0.0

        return {
            "outliers_detected": outlier_count,
            "outlier_ratio": outlier_ratio,
            "quality_score": metadata["quality_score"],
            "context_reuse_ratio": metadata["context_reuse_ratio"],
            "execution_time_ms": metadata["execution_time_ms"],
            "tier_reached": metadata["tier_reached"],
            "methods_used": metadata["methods_used"],
            "regime": metadata["regime_info"]["regime"],
            "microstructure_analysis": metadata["microstructure_analysis"],
            # NEW v2.0: Adaptive weighting metadata (FAIL-FAST for required fields)
            "adaptive_weights": metadata["adaptive_weights"],
            "decomposition_quality_score": metadata["decomposition_quality_score"],
            "residual_strength": metadata["residual_strength"],
            # Optional fields (may be absent in some scenarios)
            "quality_warnings": metadata["quality_warnings"],
            "skipped_methods": metadata["skipped_methods"],
            "fallback_used": False,
            "config_outlier_detection": self.config,
        }

    # ========== DATAFRAME ENRICHMENT ==========

    def _add_module_columns_to_dataframe(
        self, dataframe: pd.DataFrame, algorithm_result: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Add outlier detection columns to DataFrame.

        Implements DataFrame Enrichment Pattern for outlier detection.

        Args:
            dataframe: Original DataFrame
            algorithm_result: Result from algorithm

        Returns:
            DataFrame enriched with outlier columns
        """
        if algorithm_result["status"] != "success":
            logging.warning(f"{self} - Algorithm failed, skipping DataFrame enrichment")
            return dataframe

        try:
            result = algorithm_result["result"]

            # Add enrichment columns with graceful degradation
            for column in self.ENRICHMENT_COLUMNS:
                if column in result:
                    dataframe[column] = result[column]
                else:
                    logging.debug(f"{self} - Column '{column}' not in result, skipping")

            logging.info(
                f"{self} - DataFrame enriched with {len(self.ENRICHMENT_COLUMNS)} "
                f"outlier detection columns"
            )

            return dataframe

        except Exception as e:
            logging.warning(
                f"{self} - DataFrame enrichment failed: {e}, "
                f"returning original DataFrame"
            )
            return dataframe

    # ========== FALLBACK METHODS ==========

    def _simple_mode_detection(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Simple mode outlier detection using basic statistical methods.

        Fallback when main algorithm fails. Uses simple Z-score detection.

        Args:
            data: DataFrame with at least target column

        Returns:
            Dict with simple detection results
        """
        try:
            series = data[self.targetColumn]

            # Simple Z-score outlier detection
            mean = series.mean()
            std = series.std()
            z_scores = (
                (series - mean) / std if std > 0 else pd.Series(0, index=series.index)
            )
            outliers = abs(z_scores) > 3

            outliers_detected = int(outliers.sum())
            outlier_ratio = outliers_detected / len(series) if len(series) > 0 else 0.0

            logging.info(
                f"{self} - Simple mode: detected {outliers_detected} outliers "
                f"({outlier_ratio:.1%})"
            )

            return {
                "status": "success",
                "result": {
                    "outliers": outliers,
                    "outlier_confidence": pd.Series(
                        abs(z_scores) / 3, index=series.index
                    ),
                    "outlier_score_enhanced": abs(z_scores),
                    "outlier_type": pd.Series(
                        ["statistical" if o else "none" for o in outliers],
                        index=series.index,
                    ),
                    "price_robust": series.copy(),
                    "outliers_detected": outliers_detected,
                    "outlier_ratio": outlier_ratio,
                    "quality_score": 0.5,
                },
                "metadata": {
                    "mode": "simple",
                    "tier_reached": 0,
                    "methods_used": ["z_score_simple"],
                    "regime": "unknown",
                    "microstructure_analysis": {
                        "hft_noise_level": 0.0,
                        "hft_patterns": [],
                        "applicable_for_data": False,
                    },
                },
            }

        except Exception as e:
            logging.error(f"{self} - Simple mode detection failed: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Simple mode failed: {str(e)}",
            }