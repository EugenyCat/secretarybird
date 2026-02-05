"""
Time Series Analyzer Algorithm Orchestra - Protocol-Compliant Implementation.

Simple Merge strategy for combining unique analysis results from stationarity,
statistical, and outlier detection methods.
"""

import logging
from typing import Any, ClassVar, Dict, Optional, Tuple

import pandas as pd

from pipeline.helpers.utils import validate_required_locals
from pipeline.timeSeriesProcessing.analyzer.methods.outlierAnalysisMethod import (
    OutlierAnalysisMethod,
)
from pipeline.timeSeriesProcessing.analyzer.methods.stationarityMethod import (
    StationarityMethod,
)
from pipeline.timeSeriesProcessing.analyzer.methods.statisticalMethod import (
    StatisticalMethod,
)
from pipeline.timeSeriesProcessing.baseModule.baseAlgorithm import BaseAlgorithm

__version__ = "2.1.0"


class TimeSeriesAnalyzer(BaseAlgorithm):
    """
    Protocol-compliant orchestrator for time series analysis.

    STRATEGY: Simple Merge - unique results per method
    METHODS: Stationarity, Statistical, Outlier Analysis
    """

    AVAILABLE_METHODS: ClassVar[Dict[str, type]] = {
        "stationarity": StationarityMethod,
        "statistical": StatisticalMethod,
        "outlier": OutlierAnalysisMethod,
    }
    MIN_DATA_LENGTH: ClassVar[int] = 3

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize analyzer with protocol-compliant pattern."""
        validate_required_locals(["config"], locals())

        if not config:
            raise ValueError(
                "Configuration required. Use build_config_from_properties()."
            )

        self.config = config
        self.enabled_methods = config["_active_methods"]
        self.all_available_methods = config["_active_methods"]
        self._methods = {}
        self._class_name = self.__class__.__name__

        logging.info(
            f"{self._class_name} initialized: {len(self.enabled_methods)} methods"
        )

    def process(
        self, data: pd.Series, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute analysis with Simple Merge strategy.

        Returns: Standardized response with analysis results
        """
        try:
            validation = self._validate_input(data, context)
            if validation["status"] == "error":
                return validation

            execution_result = self._execute_methods(data, context)
            if execution_result["status"] == "error":
                return execution_result

            combined_result = self._combine_results(
                execution_result["method_results"]
            )
            self._add_summary_metrics(combined_result["result"])

            final_result = self._finalize_result(
                combined_result, data, context, execution_result
            )

            logging.info(
                f"{self._class_name} completed: "
                f"{len(execution_result['method_results'])} methods"
            )
            return final_result

        except Exception as e:
            return self._handle_critical_error(e)

    def _validate_input(
        self, data: pd.Series, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate input data with protocol requirements."""
        if data is None or len(data) == 0:
            return self._error_response("Input data is empty or None")

        if not isinstance(data, pd.Series):
            return self._error_response(f"Expected pd.Series, got {type(data).__name__}")

        if len(data) < self.MIN_DATA_LENGTH:
            return self._error_response(
                f"Insufficient data: {len(data)} < {self.MIN_DATA_LENGTH}",
                {"data_length": len(data)},
            )

        return {"status": "success"}

    def _execute_methods(
        self, data: pd.Series, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute all enabled methods with error handling."""
        method_results = {}
        errors = []

        for method_name in self.enabled_methods:
            if method_name not in self.AVAILABLE_METHODS:
                logging.warning(f"Unknown method: {method_name}, skipping")
                continue

            result, error = self._process_single_method(method_name, data, context)
            if result:
                method_results[method_name] = result
            if error:
                errors.append(error)

        if not method_results:
            return {
                "status": "error",
                "message": f"All {self._class_name} methods failed",
                "metadata": {
                    "algorithm": self._class_name,
                    "errors": errors,
                    "methods_attempted": self.enabled_methods,
                },
            }

        return {
            "status": "success",
            "method_results": method_results,
            "errors": errors if errors else None,
            "metadata": {
                "strategy": "simple_merge",
                "methods_attempted": self.enabled_methods,
                "methods_succeeded": list(method_results.keys()),
            },
        }

    def _process_single_method(
        self, method_name: str, data: pd.Series, context: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Process single method with error handling."""
        try:
            method = self._get_method_instance(method_name)
            result = method.process(data, context)

            if result["status"] == "success":
                return result, None

            error = {"method": method_name, "error": result.get("message", "Unknown")}
            logging.warning(f"{self._class_name} {method_name} error: {error['error']}")
            return None, error

        except Exception as e:
            logging.error(f"{self._class_name} {method_name} exception: {e}", exc_info=True)
            return None, {"method": method_name, "error": str(e)}

    def _finalize_result(
        self,
        algorithm_result: Dict[str, Any],
        data: pd.Series,
        context: Optional[Dict[str, Any]],
        execution_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add comprehensive metadata to result."""
        try:
            execution_metadata = execution_metadata or {}
            methods_used = tuple(
                execution_metadata.get("metadata", {}).get("methods_succeeded", [])
            )

            if "metadata" not in algorithm_result:
                algorithm_result["metadata"] = {}

            algorithm_result["metadata"].update({
                "algorithm": self._class_name,
                "strategy": "simple_merge",
                "methods_used": methods_used,
                "data_length": len(data),
                "interval": context.get("interval"),
                "errors": execution_metadata.get("errors"),
            })

            return algorithm_result

        except Exception as e:
            return self._handle_critical_error(e)

    def _combine_results(
        self, method_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Simple Merge - combine unique metrics from all methods."""
        result_data = {}
        method_metadata = {}

        for method_name, result in method_results.items():
            if result["status"] == "success":
                result_data.update(result["result"])
                method_metadata[method_name] = result["metadata"]

        return {
            "status": "success",
            "result": result_data,
            "metadata": {
                "combination_method": "simple_merge",
                "method_metadata": method_metadata,
            },
        }

    def _get_method_instance(self, method_name: str):
        """Lazy-load and cache method instance."""
        if method_name not in self._methods:
            if method_name not in self.AVAILABLE_METHODS:
                raise ValueError(
                    f"Unknown method: {method_name}. "
                    f"Available: {list(self.AVAILABLE_METHODS.keys())}"
                )

            method_class = self.AVAILABLE_METHODS[method_name]
            method_config = self.config[method_name]

            if not method_config:
                raise ValueError(f"Missing config for '{method_name}'")

            self._methods[method_name] = method_class(method_config)

        return self._methods[method_name]

    def _handle_critical_error(self, error: Exception) -> Dict[str, Any]:
        """Standardized error handling with logging."""
        error_msg = f"Critical error in {self._class_name}: {str(error)}"
        logging.error(error_msg, exc_info=True)

        return {
            "status": "error",
            "message": error_msg,
            "metadata": {
                "algorithm": self._class_name,
                "error_type": type(error).__name__,
                "error_stage": "algorithm_execution",
            },
        }

    def _error_response(
        self, message: str, extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create standardized error response."""
        metadata = {
            "algorithm": self._class_name,
            "validation_stage": "input_validation",
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return {"status": "error", "message": message, "metadata": metadata}

    def _add_summary_metrics(self, results: Dict[str, Any]) -> None:
        """Calculate data quality score from collected metrics."""
        missing_ratio = results["missing_ratio"]
        outlier_ratio = results["outlier_ratio"]
        noise_level = results["noise_level"]
        is_stationary = results["is_stationary"]

        quality_score = (
            (1.0 - missing_ratio)
            * (1.0 - min(outlier_ratio, 0.5))
            * (1.0 - noise_level * 0.5)
        )

        if is_stationary:
            quality_score *= 1.1

        results["data_quality_score"] = max(0.0, min(1.0, quality_score))

        logging.debug(
            f"{self._class_name} quality_score={results['data_quality_score']:.3f}"
        )

    def process_with_dataframe_enrichment(
        self,
        data: pd.DataFrame,
        target_column: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute analysis with DataFrame outlier column enrichment."""
        try:
            if data is None or data.empty:
                raise ValueError("DataFrame is empty or None")

            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found")

            if "outlier" not in self.AVAILABLE_METHODS:
                raise ValueError("OutlierAnalysisMethod not in registry")

            outlier_method_class = self.AVAILABLE_METHODS["outlier"]

            outlier_config = (
                self.config["outlier"].copy() if self.config else {}
            )
            outlier_config["return_indices"] = True

            outlier_method = outlier_method_class(outlier_config)

            enriched_df, outlier_analysis = (
                outlier_method.process_with_dataframe_enrichment(
                    data, target_column, context
                )
            )

            logging.info(f"{self._class_name} DataFrame enrichment completed")

            return enriched_df, outlier_analysis

        except Exception as e:
            logging.error(
                f"{self._class_name} enrichment error: {e}", exc_info=True
            )
            return data, context or {}

    def __str__(self) -> str:
        """Standardized string representation."""
        return f"{self._class_name}(methods={len(self.enabled_methods)})"