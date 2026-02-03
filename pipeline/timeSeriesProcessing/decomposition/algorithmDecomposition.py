"""
Decomposition Algorithm Orchestra - Protocol-Compliant Implementation.

Enhanced Decision Tree strategy for intelligent selection among 7 methods:
Fourier, SSA, TBATS, Prophet, N-BEATS, MSTL, RobustSTL.
"""

import logging
from typing import Any, ClassVar, Dict, Optional

import pandas as pd

from pipeline.helpers.configs import QualityMetricConfig
from pipeline.helpers.evaluation.qualityEvaluator import QualityEvaluator
from pipeline.helpers.utils import validate_required_locals
from pipeline.timeSeriesProcessing.baseModule.baseAlgorithm import BaseAlgorithm
from pipeline.timeSeriesProcessing.decomposition.methods.fourierDecomposerMethod import (
    FourierDecomposerMethod,
)
from pipeline.timeSeriesProcessing.decomposition.methods.mstlDecomposerMethod import (
    MSTLDecomposerMethod,
)
from pipeline.timeSeriesProcessing.decomposition.methods.nbeatsDecomposerMethod import (
    NBEATSDecomposerMethod,
)
from pipeline.timeSeriesProcessing.decomposition.methods.prophetDecomposerMethod import (
    ProphetDecomposerMethod,
)
from pipeline.timeSeriesProcessing.decomposition.methods.robustSTLDecomposerMethod import (
    RobustSTLDecomposerMethod,
)
from pipeline.timeSeriesProcessing.decomposition.methods.ssaDecomposerMethod import (
    SSADecomposerMethod,
)
from pipeline.timeSeriesProcessing.decomposition.methods.tbatsDecomposerMethod import (
    TBATSDecomposerMethod,
)

__version__ = "2.1.0"


class DecompositionAlgorithm(BaseAlgorithm):
    """
    Protocol-compliant decomposition orchestrator with Decision Tree strategy.

    Methods: Fourier, SSA, TBATS, Prophet, N-BEATS, MSTL, RobustSTL
    Hierarchy: Fourier → SSA → TBATS → Prophet → N-BEATS → MSTL → RobustSTL
    """

    AVAILABLE_METHODS: ClassVar[Dict[str, type]] = {
        "fourier": FourierDecomposerMethod,
        "ssa": SSADecomposerMethod,
        "tbats": TBATSDecomposerMethod,
        "prophet": ProphetDecomposerMethod,
        "nbeats": NBEATSDecomposerMethod,
        "mstl": MSTLDecomposerMethod,
        "robust_stl": RobustSTLDecomposerMethod,
    }
    MIN_DATA_LENGTH: ClassVar[int] = 10

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with fully adapted configuration."""
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
        self.quality_evaluator = QualityEvaluator(evaluation_type="decomposition")
        self.baseline_result = None

        logging.info(
            f"{self} initialized with {len(self.all_available_methods)} methods"
        )

    def process(
        self, data: pd.Series, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute decomposition with Decision Tree method selection."""
        try:
            # Validation
            validation = self._validate_input(data, context)
            if validation["status"] == "error":
                return validation

            # STL baseline for diagnostics
            logging.info(f"{self} Executing STL baseline")
            self.baseline_result = self._run_stl_baseline(data, context)
            if self.baseline_result["status"] == "error":
                return {
                    "status": "error",
                    "message": "STL baseline failed - decomposition impossible",
                    "metadata": {"algorithm": self._class_name},
                }

            baseline_quality = self.baseline_result["quality_score"]

            # Method selection
            predetermined = self.config.get("_predetermined_method")
            if predetermined:
                selected_method = predetermined
                selection_reason = f"predetermined_{predetermined}"
                logging.info(f"{self} Using predetermined: {predetermined}")
            else:
                selected_method = self._combine_results(
                    data, context, self.baseline_result
                )
                selection_reason = f"decision_tree_{selected_method}"
                logging.info(f"{self} Decision tree selected: {selected_method}")

            # Execute method
            method_execution = self._execute_methods(data, context, selected_method)
            if method_execution["status"] == "error":
                logging.warning(
                    f"Method {selected_method} failed, using STL fallback"
                )
                decomposition_result = self._prepare_baseline_as_fallback()
            else:
                decomposition_result = method_execution["method_results"][
                    selected_method
                ]

            # Finalize
            selection_metadata = {
                "selected_method": selected_method,
                "baseline_quality": baseline_quality,
                "selection_reason": selection_reason,
            }
            final_result = self._finalize_result(
                decomposition_result, data, context, selection_metadata=selection_metadata
            )

            logging.info(f"Decomposition completed: {selected_method}")
            return final_result

        except Exception as e:
            return self._handle_critical_error(e)

    def _validate_input(
        self, data: pd.Series, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate input data and minimum length."""
        if data is None or len(data) == 0:
            return self._error_response("Input data empty or None")

        if not isinstance(data, pd.Series):
            return self._error_response(
                f"Expected pd.Series, got {type(data).__name__}"
            )

        if len(data) < self.MIN_DATA_LENGTH:
            return self._error_response(
                f"Insufficient data: {len(data)} < {self.MIN_DATA_LENGTH}"
            )

        return {"status": "success"}

    def _execute_methods(
        self, data: pd.Series, context: Optional[Dict[str, Any]], selected_method: str
    ) -> Dict[str, Any]:
        """Execute selected method (Decision Tree executes only one)."""
        try:
            method = self._get_method_instance(selected_method)
            result = method.process(data, context)

            return {
                "status": result["status"],
                "method_results": {selected_method: result},
                "metadata": {
                    "strategy": "decision_tree",
                    "methods_attempted": [selected_method],
                    "methods_succeeded": [selected_method]
                    if result["status"] == "success"
                    else [],
                },
            }

        except Exception as e:
            logging.error(f"{self} Error in {selected_method}: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error executing {selected_method}: {str(e)}",
                "metadata": {
                    "strategy": "decision_tree",
                    "methods_attempted": [selected_method],
                    "error_method": selected_method,
                },
            }

    def _finalize_result(
        self,
        algorithm_result: Dict[str, Any],
        data: pd.Series,
        context: Optional[Dict[str, Any]],
        selection_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add quality metrics and comprehensive metadata."""
        try:
            # Quality evaluation
            decomposition = algorithm_result["result"]
            trend, seasonal, residual = (
                decomposition["trend"],
                decomposition["seasonal"],
                decomposition["residual"],
            )

            if all(c is not None for c in [trend, seasonal, residual]):
                quality_scores = self.quality_evaluator.evaluate_decomposition(
                    data, trend, seasonal, residual
                )
                algorithm_result["result"]["quality_metrics"] = quality_scores
                algorithm_result["result"]["quality_score"] = quality_scores[
                    "composite_score"
                ]

            # Extract metadata
            selection_metadata = selection_metadata or {}
            baseline_quality = selection_metadata.get("baseline_quality", 0.0)
            is_fallback = algorithm_result.get("metadata", {}).get(
                "fallback_used", False
            )

            ts_id = (
                context["currentProperties"].get("ts_id", "unknown")
                if context and "currentProperties" in context
                else "unknown"
            )

            # Determine method/reason based on fallback
            if is_fallback:
                selected_method = algorithm_result["metadata"]["selected_method"]
                original = selection_metadata.get("selected_method", "unknown")
                selection_reason = f"fallback_after_{original}_failed"
                fallback_data = {
                    k: algorithm_result["metadata"][k]
                    for k in ["fallback_reason", "baseline_reused"]
                    if k in algorithm_result["metadata"]
                }
            else:
                selected_method = selection_metadata["selected_method"]
                selection_reason = selection_metadata["selection_reason"]
                fallback_data = {}

            # Update metadata
            if "metadata" not in algorithm_result:
                algorithm_result["metadata"] = {}

            algorithm_result["metadata"].update(
                {
                    "algorithm": self._class_name,
                    "selected_method": selected_method,
                    "selection_reason": selection_reason,
                    "baseline_quality": baseline_quality,
                    "data_length": len(data),
                    "ts_id": ts_id,
                    "strategy": "decision_tree",
                    **fallback_data,
                }
            )

            return algorithm_result

        except Exception as e:
            return self._handle_critical_error(e)

    def _combine_results(
        self,
        data: pd.Series,
        context: Optional[Dict[str, Any]],
        baseline_result: Dict[str, Any],
    ) -> str:
        """
        Enhanced Decision Tree method selection.

        Hierarchy: Fourier → SSA → TBATS → Prophet → N-BEATS → MSTL → RobustSTL
        Based on: stationarity, periodicity, noise, autocorrelation, volatility
        """
        data_length = len(data)
        char = self._extract_characteristics_unified(context, data_length)
        baseline_quality = baseline_result["quality_score"]

        logging.info(
            f"Decision Tree: len={data_length}, baseline={baseline_quality:.3f}"
        )

        # 1. FOURIER - stationary with clear periodicity
        if (
            char["is_stationary"]
            and char["noise_level"] < 0.05
            and char["periodicity_quality"] > 0.7
            and "fourier" in self.all_available_methods
        ):
            return self._log_and_select(
                "fourier",
                f"stationary+clear_periodicity (noise={char['noise_level']:.3f})",
            )

        # 2. SSA - non-stationary with high autocorr or noise
        if "ssa" in self.all_available_methods:
            if (
                not char["is_stationary"]
                and char["lag1_autocorr"]
                and char["lag1_autocorr"] > 0.95
            ):
                return self._log_and_select(
                    "ssa", f"high_autocorr ({char['lag1_autocorr']:.4f})"
                )
            elif (
                not char["is_stationary"]
                and char["noise_level"] > 0.2
                and char["data_quality"] < 0.7
            ):
                return self._log_and_select(
                    "ssa", f"noisy_nonstationary (noise={char['noise_level']:.3f})"
                )

        # 3. TBATS - multiple seasonality with Box-Cox
        if (
            "tbats" in self.all_available_methods
            and len(char["periods"]) > 2
            and char["volatility"] > 0.3
            and (char["skewness"] > 1.0 or char["volatility"] > 0.5)
        ):
            return self._log_and_select(
                "tbats",
                f"complex_seasonality ({len(char['periods'])} periods, vol={char['volatility']:.3f})",
            )

        # 4. PROPHET - financial trends with changepoints
        if "prophet" in self.all_available_methods:
            prophet_optimal = (
                char["trend_strength"] > 0.25
                and char["missing_ratio"] > 0.02
                and char["instrument_type"] == "crypto"
            )
            prophet_strong = char["trend_strength"] > 0.4

            if prophet_optimal:
                return self._log_and_select(
                    "prophet", f"financial_trends (trend={char['trend_strength']:.3f})"
                )
            elif prophet_strong:
                return self._log_and_select(
                    "prophet", f"strong_trend ({char['trend_strength']:.3f})"
                )

        # 5. N-BEATS - complex nonlinear patterns
        if "nbeats" in self.all_available_methods:
            nbeats_optimal = (
                data_length > 1000
                and baseline_quality < 0.4
                and char["data_quality"] > 0.8
            )
            nbeats_nonlinear = char["kurtosis"] > 3.0 and data_length > 1000

            if nbeats_optimal:
                return self._log_and_select(
                    "nbeats", f"nonlinearity_paradox (baseline={baseline_quality:.3f})"
                )
            elif nbeats_nonlinear:
                return self._log_and_select(
                    "nbeats", f"heavy_tails (kurtosis={char['kurtosis']:.3f})"
                )

        # 6. MSTL - multiple seasonality fallback
        if (
            len(char["periods"]) > 1
            and char["periodicity_quality"] > 0.6
            and "mstl" in self.all_available_methods
        ):
            return self._log_and_select(
                "mstl", f"multiple_periods ({len(char['periods'])})"
            )

        # 7. ROBUST STL - volatility/outliers
        if (
            char["volatility"] > 0.7 or char["outlier_ratio"] > 0.15
        ) and "robust_stl" in self.all_available_methods:
            return self._log_and_select(
                "robust_stl", f"high_volatility ({char['volatility']:.3f})"
            )

        # 8. Fourier second chance (relaxed)
        if (
            char["detection_status"] in ["high_confidence", "detected"]
            and char["main_period"] > 2
            and char["data_quality"] > 0.7
            and baseline_quality > 0.6
            and char["periodicity_quality"] > 0.8
            and char["noise_level"] < 0.1
            and "fourier" in self.all_available_methods
        ):
            return self._log_and_select("fourier", "relaxed_criteria+good_data")

        # 9. Poor data quality
        if char["data_quality"] < 0.5:
            return self._log_and_select(
                "robust_stl", f"poor_quality ({char['data_quality']:.3f})"
            )

        # 10. Weak periodicity detection
        if (
            char["detection_status"] in ["low_confidence", "spurious", "not_detected"]
            and "mstl" in self.all_available_methods
        ):
            return self._log_and_select(
                "mstl", f"hidden_patterns ({char['detection_status']})"
            )

        # 11. Universal fallback
        return self._log_and_select("robust_stl", "universal_default")

    def _get_method_instance(self, method_name: str):
        """Lazy-loading with caching."""
        if method_name not in self._methods:
            if method_name not in self.AVAILABLE_METHODS:
                raise ValueError(
                    f"Unknown method: {method_name}. "
                    f"Available: {list(self.AVAILABLE_METHODS.keys())}"
                )

            method_class = self.AVAILABLE_METHODS[method_name]
            method_config = self.config[method_name]

            if not method_config:
                raise ValueError(
                    f"Missing config for '{method_name}'. "
                    f"Ensure config contains all method settings."
                )

            self._methods[method_name] = method_class(method_config)

        return self._methods[method_name]

    def _handle_critical_error(self, error: Exception) -> Dict[str, Any]:
        """Standardized error handling."""
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

    def _run_stl_baseline(
        self, data: pd.Series, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute STL baseline for diagnostics and fallback."""
        try:
            robust_stl = self._get_method_instance("robust_stl")
            result = robust_stl.process(data, context)

            if result["status"] == "success":
                decomp = result["result"]
                if all(
                    decomp[k] is not None for k in ["trend", "seasonal", "residual"]
                ):
                    quality = (
                        self.quality_evaluator.evaluate_decomposition_single_metric(
                            QualityMetricConfig.SEASONAL_STRENGTH,
                            data,
                            decomp["trend"],
                            decomp["seasonal"],
                            decomp["residual"],
                        )
                    )
                    result["quality_score"] = quality

            return result

        except Exception as e:
            return {"status": "error", "message": f"STL baseline error: {e}"}

    def _extract_characteristics_unified(
        self, context: Optional[Dict[str, Any]], data_length: int
    ) -> Dict[str, Any]:
        """Extract characteristics from context for decision tree."""
        if not context or "currentProperties" not in context:
            raise ValueError(
                f"{self._class_name} Missing context['currentProperties']. "
                f"Ensure analyzer and periodicity executed first."
            )

        props = context["currentProperties"]

        if "analyzer" not in props or not props["analyzer"]:
            raise ValueError(
                f"{self._class_name} Missing analyzer properties. "
                f"Analyzer must execute before decomposition."
            )

        if "periodicity" not in props or not props["periodicity"]:
            raise ValueError(
                f"{self._class_name} Missing periodicity properties. "
                f"Periodicity must execute before decomposition."
            )

        a = props["analyzer"]
        p = props["periodicity"]

        return {
            "volatility": a.get("volatility", 0.0),
            "outlier_ratio": a.get("outlier_ratio", 0.0),
            "data_quality": a.get("data_quality_score", 0.0),
            "is_stationary": a.get("is_stationary", False),
            "noise_level": a.get("noise_level", 0.0),
            "lag1_autocorr": a.get("lag1_autocorrelation"),
            "skewness": a.get("skewness", 0.0),
            "kurtosis": a.get("kurtosis", 0.0),
            "trend_strength": a.get("estimated_trend_strength", 0.0),
            "missing_ratio": a.get("missing_ratio", 0.0),
            "main_period": self.config["_all_periods"][0],
            "periods": self.config["_all_periods"],
            "periodicity_quality": p.get("periodicity_quality_score", 0.0),
            "detection_status": p.get("detection_status", "not_detected"),
            "instrument_type": props.get("instrument_type", "unknown"),
            "data_length": data_length,
        }

    def _prepare_baseline_as_fallback(self) -> Dict[str, Any]:
        """Prepare cached baseline as fallback."""
        try:
            if not hasattr(self, "baseline_result") or not self.baseline_result:
                return self._error_response("STL baseline unavailable for fallback")

            if self.baseline_result["status"] != "success":
                return self._error_response(
                    "STL baseline unsuccessful - cannot use as fallback"
                )

            fallback = self.baseline_result.copy()
            if "metadata" not in fallback:
                fallback["metadata"] = {}

            fallback["metadata"].update(
                {
                    "fallback_used": True,
                    "fallback_reason": "primary_method_failed",
                    "selected_method": "robust_stl",
                    "baseline_reused": True,
                }
            )

            logging.info(f"{self} STL baseline reused as fallback")
            return fallback

        except Exception as e:
            return self._handle_critical_error(e)

    def _error_response(self, message: str) -> Dict[str, Any]:
        """Create error response helper."""
        return {
            "status": "error",
            "message": message,
            "metadata": {
                "algorithm": self._class_name,
                "validation_stage": "input_validation",
            },
        }

    def _log_and_select(self, method: str, reason: str) -> str:
        """Helper for logging method selection."""
        logging.info(f"{self} Decision Tree: {method} - {reason}")
        return method

    def __str__(self) -> str:
        """Standard string representation."""
        return f"{self._class_name}(methods={len(self.all_available_methods)})"