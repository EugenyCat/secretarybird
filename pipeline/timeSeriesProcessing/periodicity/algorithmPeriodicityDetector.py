"""
Periodicity Detection Algorithm Orchestra - Protocol-Compliant Implementation.

Implements Consensus Weighted Voting strategy for ensemble periodicity detection
using ACF, Spectral Analysis, and Wavelet methods.
"""

import logging
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.helpers.utils import validate_required_locals
from pipeline.timeSeriesProcessing.baseModule.baseAlgorithm import BaseAlgorithm
from pipeline.timeSeriesProcessing.periodicity.methods.acfMethod import ACFMethod
from pipeline.timeSeriesProcessing.periodicity.methods.spectralMethod import (
    SpectralMethod,
)
from pipeline.timeSeriesProcessing.periodicity.methods.waveletMethod import (
    WaveletMethod,
)

__version__ = "2.1.0"


class PeriodicityDetector(BaseAlgorithm):
    """
    Protocol-compliant orchestrator for periodicity detection methods.

    STRATEGY: Consensus Weighted Voting for ensemble detection
    METHODS: ACF, Spectral Analysis, Wavelet Transform
    """

    AVAILABLE_METHODS: ClassVar[Dict[str, type]] = {
        "acf": ACFMethod,
        "spectral": SpectralMethod,
        "wavelet": WaveletMethod,
    }
    MIN_DATA_LENGTH: ClassVar[int] = 6

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize periodicity detector following protocol pattern."""
        validate_required_locals(["config"], locals())

        if not config:
            raise ValueError(
                "Configuration required. Use build_config_from_properties()."
            )

        self.config = config
        self.enabled_methods = self.config["_active_methods"]
        self.all_available_methods = self.config["_active_methods"]
        self._methods = {}
        self._class_name = self.__class__.__name__

        logging.info(
            f"{self._class_name} initialized: {len(self.enabled_methods)} methods"
        )

    def process(
        self, data: pd.Series, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute periodicity detection with Consensus Weighted Voting.

        Workflow: validate → execute → combine → finalize
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
                combined_result, data, context, execution_metadata=execution_result
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
        """Validate input data following protocol requirements."""
        if data is None or len(data) == 0:
            return {
                "status": "error",
                "message": "Input data is empty or None",
                "metadata": {"algorithm": self._class_name},
            }

        if not isinstance(data, pd.Series):
            return {
                "status": "error",
                "message": f"Expected pd.Series, got {type(data).__name__}",
                "metadata": {"algorithm": self._class_name},
            }

        if len(data) < self.MIN_DATA_LENGTH:
            return {
                "status": "error",
                "message": f"Insufficient data: {len(data)} < {self.MIN_DATA_LENGTH}",
                "metadata": {"algorithm": self._class_name, "data_length": len(data)},
            }

        return {"status": "success"}

    def _execute_methods(
        self, data: pd.Series, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute all enabled periodicity detection methods."""
        method_results, errors = {}, []

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
                "strategy": "consensus",
                "methods_attempted": self.enabled_methods,
                "methods_succeeded": list(method_results.keys()),
            },
        }

    def _process_single_method(
        self, method_name: str, data: pd.Series, context: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Process single periodicity detection method with error handling."""
        try:
            method = self._get_method_instance(method_name)
            result = method.process(data, context)

            if result["status"] == "success":
                return result, None

            error = {"method": method_name, "error": result.get("message", "Unknown")}
            logging.warning(f"{self._class_name} {method_name}: {error['error']}")
            return None, error

        except Exception as e:
            logging.error(f"{self._class_name} {method_name}: {e}", exc_info=True)
            return None, {"method": method_name, "error": str(e)}

    def _finalize_result(
        self,
        algorithm_result: Dict[str, Any],
        data: pd.Series,
        context: Optional[Dict[str, Any]],
        execution_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Finalize detection result with comprehensive metadata."""
        try:
            execution_metadata = execution_metadata or {}
            methods_used = tuple(
                execution_metadata.get("metadata", {}).get("methods_succeeded", [])
            )

            result = algorithm_result["result"]
            if "metadata" not in algorithm_result:
                algorithm_result["metadata"] = {}

            algorithm_result["metadata"].update(
                {
                    "algorithm": self._class_name,
                    "strategy": "consensus",
                    "methods_used": methods_used,
                    "data_length": len(data),
                    "interval": context.get("interval") if context else None,
                    "errors": execution_metadata.get("errors"),
                    "consensus_strength": result.get("consensus_strength", 0.0),
                    "validation_score": result.get("validation_score", 0.0),
                }
            )

            return algorithm_result

        except Exception as e:
            return self._handle_critical_error(e)

    def _combine_results(
        self, method_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Consensus Weighted Voting strategy for ensemble detection.

        Algorithm: extract → group → evaluate → validate → select
        """
        all_periods, method_metadata = [], {}

        for method_name, result in method_results.items():
            if result["status"] == "success":
                method_result = result["result"]
                periods = method_result.get("periods", [])
                confidences = method_result.get("confidence_scores", [])

                for period, conf in zip(periods, confidences):
                    if period > 0:
                        all_periods.append(
                            {"period": period, "confidence": conf, "method": method_name}
                        )

                method_metadata[method_name] = result["metadata"]

        if all_periods:
            period_groups = self._group_similar_periods_advanced(all_periods)
            evaluated_groups = self._evaluate_period_groups(period_groups)
            best_group = self._select_best_period_group(evaluated_groups)

            if best_group:
                main_period = best_group["consensus_period"]
                main_confidence = best_group["consensus_confidence"]
                consensus_strength = best_group["consensus_strength"]
                validation_score = self._cross_validate_period(main_period, all_periods)

                all_unique_periods, all_confidences = [], []
                for group in evaluated_groups:
                    if group["consensus_period"] not in all_unique_periods:
                        all_unique_periods.append(group["consensus_period"])
                        all_confidences.append(group["consensus_confidence"])

                if len(all_unique_periods) > 10:
                    sorted_pairs = sorted(
                        zip(all_unique_periods, all_confidences),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:10]
                    all_unique_periods = [p[0] for p in sorted_pairs]
                    all_confidences = [p[1] for p in sorted_pairs]
            else:
                (
                    main_period,
                    main_confidence,
                    consensus_strength,
                    validation_score,
                ) = (0, 0.0, 0.0, 0.0)
                all_unique_periods, all_confidences = [], []
        else:
            (main_period, main_confidence, consensus_strength, validation_score) = (
                0,
                0.0,
                0.0,
                0.0,
            )
            all_unique_periods, all_confidences = [], []

        detection_status = self._get_consensus_status(
            main_confidence, main_period, consensus_strength, validation_score
        )

        return {
            "status": "success",
            "result": {
                "main_period": main_period,
                "main_confidence": main_confidence,
                "periods": all_unique_periods,
                "confidence_scores": all_confidences,
                "detection_method": "consensus_ensemble",
                "detection_status": detection_status,
                "consensus_strength": consensus_strength,
                "validation_score": validation_score,
                "total_candidates": len(all_periods),
            },
            "metadata": {
                "combination_method": "consensus_weighted_voting",
                "method_metadata": method_metadata,
            },
        }

    def _get_method_instance(self, method_name: str):
        """Lazy-loading method instantiation with caching."""
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
                    f"Missing configuration for '{method_name}'. "
                    f"Ensure config contains settings for all methods."
                )

            self._methods[method_name] = method_class(method_config)

        return self._methods[method_name]

    def _handle_critical_error(self, error: Exception) -> Dict[str, Any]:
        """Standardized critical error handling following protocol."""
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

    def _group_similar_periods_advanced(
        self, all_periods: List[Dict]
    ) -> List[List[Dict]]:
        """
        Advanced grouping with adaptive thresholds.

        Groups statistically similar periods based on magnitude-dependent thresholds.
        """
        if not all_periods:
            return []

        sorted_periods = sorted(all_periods, key=lambda x: x["period"])
        groups = []

        for period_item in sorted_periods:
            current_period = period_item["period"]
            assigned = False

            for group in groups:
                threshold = self._calculate_adaptive_threshold(current_period)

                for group_item in group:
                    relative_diff = abs(current_period - group_item["period"]) / max(
                        current_period, group_item["period"]
                    )

                    if relative_diff <= threshold:
                        group.append(period_item)
                        assigned = True
                        break

                if assigned:
                    break

            if not assigned:
                groups.append([period_item])

        return groups

    def _calculate_adaptive_threshold(self, period: float) -> float:
        """Calculate adaptive similarity threshold based on period magnitude."""
        if period < 10:
            return 0.05  # 5% for small
        elif period < 100:
            return 0.08  # 8% for medium
        return 0.10  # 10% for large

    def _evaluate_period_groups(self, period_groups: List[List[Dict]]) -> List[Dict]:
        """
        Evaluate groups with weighted voting and consensus scoring.

        Calculates: consensus period, confidence (with multi-method bonus), strength.
        """
        evaluated_groups = []

        for group in period_groups:
            if not group:
                continue

            total_weight = sum(item["confidence"] for item in group)
            consensus_period = (
                sum(item["period"] * item["confidence"] for item in group) / total_weight
                if total_weight > 0
                else float(np.mean([item["period"] for item in group]))
            )

            base_confidence = float(np.mean([item["confidence"] for item in group]))
            unique_methods = set(item["method"] for item in group)
            method_count = len(unique_methods)

            consensus_bonus = 1.0 + (method_count - 1) * 0.15
            consensus_confidence = min(1.0, base_confidence * consensus_bonus)
            consensus_strength = self._calculate_consensus_strength(group, method_count)

            evaluated_groups.append(
                {
                    "consensus_period": round(consensus_period, 2),
                    "consensus_confidence": consensus_confidence,
                    "consensus_strength": consensus_strength,
                    "method_count": method_count,
                    "contributing_methods": list(unique_methods),
                    "group_size": len(group),
                    "original_items": group,
                }
            )

        evaluated_groups.sort(
            key=lambda x: x["consensus_confidence"] * x["consensus_strength"],
            reverse=True,
        )

        return evaluated_groups

    def _calculate_consensus_strength(
        self, group: List[Dict], method_count: int
    ) -> float:
        """Calculate consensus strength: method count + confidence stability."""
        base_strength = min(1.0, method_count / 3.0)

        confidences = [item["confidence"] for item in group]
        confidence_std = float(np.std(confidences)) if len(confidences) > 1 else 0.0
        stability_factor = max(0.5, 1.0 - confidence_std)

        return base_strength * stability_factor

    def _select_best_period_group(self, evaluated_groups: List[Dict]) -> Optional[Dict]:
        """Select best period group with false positive prevention."""
        if not evaluated_groups:
            return None

        valid_groups = [
            g
            for g in evaluated_groups
            if g["consensus_strength"] >= 0.3
            and not (g["method_count"] == 1 and g["consensus_confidence"] < 0.8)
        ]

        return valid_groups[0] if valid_groups else None

    def _cross_validate_period(self, period: float, all_periods: List[Dict]) -> float:
        """Cross-validate period through statistical stability tests."""
        if period <= 0 or not all_periods:
            return 0.0

        threshold = self._calculate_adaptive_threshold(period)
        related_detections = [
            item
            for item in all_periods
            if abs(period - item["period"]) / max(period, item["period"]) <= threshold
        ]

        if len(related_detections) < 2:
            return 0.5

        period_values = [item["period"] for item in related_detections]
        confidence_values = [item["confidence"] for item in related_detections]

        period_cv = (
            float(np.std(period_values)) / float(np.mean(period_values))
            if np.mean(period_values) > 0
            else 1.0
        )
        stability_score = max(0.0, 1.0 - period_cv * 5)
        avg_confidence = float(np.mean(confidence_values))

        validation_score = stability_score * 0.6 + avg_confidence * 0.4
        return min(1.0, max(0.0, validation_score))

    def _get_consensus_status(
        self,
        confidence: float,
        period: int,
        consensus_strength: float,
        validation_score: float,
    ) -> str:
        """Determine detection status based on consensus quality."""
        if period == 0:
            return "not_detected"

        quality_score = (
            confidence * 0.4 + consensus_strength * 0.4 + validation_score * 0.2
        )

        if quality_score >= 0.75:
            return "high_confidence"
        elif quality_score >= 0.55:
            return "detected"
        elif quality_score >= 0.35:
            return "low_confidence"
        return "spurious"

    def _add_summary_metrics(self, results: Dict[str, Any]) -> None:
        """Calculate and add summary quality metrics to detection results."""
        main_confidence = results.get("main_confidence", 0.0)
        periods_count = len(results.get("periods", []))

        detection_quality = main_confidence * (1.0 + min(periods_count, 5) * 0.1)
        results["periodicity_quality_score"] = max(0.0, min(1.0, detection_quality))

        logging.debug(
            f"{self._class_name} quality_score="
            f"{results['periodicity_quality_score']:.3f}"
        )

    def __str__(self) -> str:
        """Standardized string representation for logging."""
        return f"{self._class_name}(methods={len(self.enabled_methods)})"