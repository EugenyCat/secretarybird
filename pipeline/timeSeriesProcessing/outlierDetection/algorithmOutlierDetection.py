"""
OutlierDetectionAlgorithm - Level 2 Algorithm with Enhanced Decision Tree.

Implements BaseAlgorithm Protocol with tiered execution:
- Tier 1: StatisticalEnhancementMethod (always executed)
- Tier 2: ComponentAnomalyMethod (conditional, quality-based)
- Early stopping, adaptive weighting, regime classification
- Financial helpers: regime classification, microstructure analysis

Architecture: Processor → Algorithm (THIS) → Methods (Tier 1-2) → Helpers
Version: 2.2.0 (Adaptive Weighting + Residual Consensus)
"""

import logging
import time
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.helpers.utils import validate_required_locals
from pipeline.timeSeriesProcessing.outlierDetection.methods.baseOutlierDetectionMethod import (
    BaseOutlierDetectionMethod,
)
from pipeline.timeSeriesProcessing.outlierDetection.methods.componentAnomalyMethod import (
    ComponentAnomalyMethod,
)
from pipeline.timeSeriesProcessing.outlierDetection.methods.statisticalEnhancementMethod import (
    StatisticalEnhancementMethod,
)

__version__ = "2.2.0"


class OutlierDetectionAlgorithm:
    """
    Level 2 Algorithm: Enhanced Decision Tree with Tiered Execution (v2.2.0).
    Implements BaseAlgorithm Protocol with adaptive weighting and quality-based method selection.
    """

    AVAILABLE_METHODS: ClassVar[Dict[str, type]] = {
        "statistical_enhancement": StatisticalEnhancementMethod,
        "component_anomaly": ComponentAnomalyMethod,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize algorithm with BaseAlgorithm Protocol compliance."""
        validate_required_locals(["config"], locals())
        if config is None:
            raise ValueError("OutlierDetectionAlgorithm requires non-None config")

        self.config = config
        self.enabled_methods = self.config["_active_methods"]
        self.all_available_methods = list(self.AVAILABLE_METHODS.keys())
        self._methods: Dict[str, Any] = {}
        self._class_name = self.__class__.__name__

        self.early_stopping_threshold = self.config.get(
            "early_stopping_threshold", 0.95
        )
        self.tier2_quality_threshold = self.config.get(
            "tier2_decomposition_quality_threshold", 0.5
        )
        self.enable_regime_classification = self.config.get(
            "enable_regime_classification", True
        )

        logging.info(
            f"{self} - Initialized: active_methods={self.enabled_methods}, "
            f"early_stopping={self.early_stopping_threshold:.2f}"
        )

    def __str__(self) -> str:
        """Standardized string representation for logging."""
        return f"{self._class_name}(methods={len(self.enabled_methods)})"

    def process(
        self, data: pd.DataFrame, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Main algorithm execution: validate → execute → combine → finalize."""
        try:
            validation_result = self._validate_input(data, context)
            if validation_result["status"] != "success":
                return validation_result

            execution_result = self._execute_methods(data, context)
            if execution_result["status"] != "success":
                return self._finalize_result(execution_result, data, context)

            combined_result = self._combine_results_enhanced_decision_tree(
                execution_result, data, context
            )
            final_result = self._finalize_result(combined_result, data, context)

            logging.info(
                f"{self} - Completed: outliers={final_result['result']['outlier_count']}, tier={final_result['metadata']['tier_reached']}"
            )

            return final_result

        except Exception as e:
            return self._handle_critical_error(e)

    def _validate_input(
        self, data: pd.DataFrame, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate input data and required columns."""
        if data is None or len(data) == 0:
            return self._error_response("Input data empty or None")

        if not isinstance(data, pd.DataFrame):
            return self._error_response(
                f"Expected pd.DataFrame, got {type(data).__name__}"
            )

        if len(data) < self.config["base"]["min_data_length"]:
            return self._error_response(
                f"Data length {len(data)} < minimum {self.config['base']['min_data_length']}"
            )

        required = ["is_zscore_outlier", "is_iqr_outlier", "is_mad_outlier"]
        missing = [col for col in required if col not in data.columns]
        if missing:
            return self._error_response(f"Missing required columns: {missing}")

        if context is None:
            logging.warning(f"{self} - Context is None")

        return {"status": "success"}

    def _execute_methods(
        self, data: pd.DataFrame, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute tiered methods with early stopping."""
        start_time = time.perf_counter()

        try:
            selected_methods, regime_info = self._analyze_context_and_select_methods(
                context
            )

            method_results = {}
            methods_succeeded = []
            tier_reached = 0

            # Execute Tier 1 (always)
            if "statistical_enhancement" in selected_methods:
                tier1_method = self._get_method_instance("statistical_enhancement")
                tier1_result = tier1_method.detect(data, context)

                if tier1_result["status"] == "success":
                    method_results["statistical_enhancement"] = tier1_result
                    methods_succeeded.append("statistical_enhancement")
                    tier_reached = 1

                    # Early stopping check
                    if (
                        tier1_result["mean_consensus_strength"]
                        > self.early_stopping_threshold
                    ):
                        execution_time = (time.perf_counter() - start_time) * 1000
                        logging.info(
                            f"{self} - Early stopping at Tier 1 (confidence={tier1_result['mean_consensus_strength']:.3f})"
                        )
                        return {
                            "status": "success",
                            "method_results": method_results,
                            "metadata": {
                                "strategy": "enhanced_decision_tree",
                                "methods_attempted": selected_methods[:1],
                                "methods_succeeded": methods_succeeded,
                                "tier_reached": tier_reached,
                                "regime_info": regime_info,
                                "execution_time_ms": execution_time,
                            },
                        }

            # Execute Tier 2 (conditional)
            if "component_anomaly" in selected_methods:
                tier2_method = self._get_method_instance("component_anomaly")
                tier2_result = tier2_method.detect(data, context)

                if tier2_result["status"] == "success":
                    method_results["component_anomaly"] = tier2_result
                    methods_succeeded.append("component_anomaly")
                    tier_reached = 2

            execution_time = (time.perf_counter() - start_time) * 1000

            return {
                "status": "success",
                "method_results": method_results,
                "metadata": {
                    "strategy": "enhanced_decision_tree",
                    "methods_attempted": selected_methods,
                    "methods_succeeded": methods_succeeded,
                    "tier_reached": tier_reached,
                    "regime_info": regime_info,
                    "execution_time_ms": execution_time,
                },
            }

        except Exception as e:
            logging.error(f"{self} - Method execution failed: {e}", exc_info=True)
            return self._error_response(f"Method execution failed: {str(e)}")

    def _finalize_result(
        self,
        algorithm_result: Dict[str, Any],
        data: pd.DataFrame,
        context: Optional[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        """Finalize result with microstructure analysis and v2.0 metadata."""
        if algorithm_result["status"] == "error":
            return algorithm_result

        result = algorithm_result["result"]
        metadata = algorithm_result["metadata"]

        microstructure_analysis = self._detect_microstructure_issues(data, context)

        finalized_metadata = {
            "algorithm": self._class_name,
            "data_length": len(data),
            "methods_used": metadata["methods_succeeded"],
            "strategy": "enhanced_decision_tree",
            "tier_reached": metadata["tier_reached"],
            "execution_time_ms": metadata["execution_time_ms"],
            "context_reuse_ratio": metadata["context_reuse_ratio"],
            "quality_score": metadata["quality_score"],
            "regime_info": metadata["regime_info"],
            "microstructure_analysis": microstructure_analysis,
            "adaptive_weights": metadata["adaptive_weights"],
            "decomposition_quality_score": metadata["decomposition_quality_score"],
            "residual_strength": metadata["residual_strength"],
            "quality_warnings": metadata["quality_warnings"],
            "skipped_methods": metadata["skipped_methods"],
        }

        return {"status": "success", "result": result, "metadata": finalized_metadata}

    def _get_method_instance(self, method_name: str):
        """Lazy-loading method with caching."""
        if method_name not in self._methods:
            if method_name not in self.AVAILABLE_METHODS:
                raise ValueError(f"Unknown method: {method_name}")

            method_class = self.AVAILABLE_METHODS[method_name]
            self._methods[method_name] = method_class(self.config[method_name])
            logging.debug(f"{self} - Initialized: {method_name}")

        return self._methods[method_name]

    def _handle_critical_error(self, error: Exception) -> Dict[str, Any]:
        """Standardized error handling."""
        logging.error(
            f"Critical error in {self._class_name}: {str(error)}", exc_info=True
        )
        return {
            "status": "error",
            "message": f"Critical error in {self._class_name}: {str(error)}",
            "metadata": {
                "algorithm": self._class_name,
                "error_type": type(error).__name__,
            },
        }

    def _error_response(self, message: str) -> Dict[str, Any]:
        """Create error response helper."""
        return {
            "status": "error",
            "message": message,
            "metadata": {"algorithm": self._class_name},
        }

    def _analyze_context_and_select_methods(
        self, context: Optional[Dict[str, Any]]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Analyze context and select methods via Enhanced Decision Tree."""
        selected_methods = ["statistical_enhancement"]
        regime_info = self._classify_regime(context)

        if context:
            try:
                quality_score = context["currentProperties"]["decomposition"][
                    "quality_score"
                ]
                if quality_score > self.tier2_quality_threshold:
                    selected_methods.append("component_anomaly")
            except (KeyError, AttributeError) as e:
                logging.warning(f"{self} - Context analysis failed: {e}")

        return selected_methods, regime_info

    def _classify_regime(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Classify market regime using financial.regime helper."""
        from pipeline.helpers.financial.regime import classify_market_regime

        if not context:
            raise ValueError("Context required for regime classification")

        analyzer = context["currentProperties"]["analyzer"]
        helper_result = classify_market_regime(analyzer)

        if helper_result["status"] != "success":
            raise RuntimeError(
                f"Regime classification failed: {helper_result.get('message')}"
            )

        result = helper_result["result"]
        return {
            "regime": result["market_regime"],
            "confidence": result["regime_confidence"],
            "volatility": float(analyzer["volatility"]),
            "trend_strength": float(analyzer["estimated_trend_strength"]),
            "volatility_regime": result["volatility_regime"],
            "trend_regime": result["trend_regime"],
            "persistence_regime": result["persistence_regime"],
            "helper_used": True,
        }

    def _detect_microstructure_issues(
        self, data: pd.DataFrame, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Detect HFT microstructure patterns using financial.microstructure helper."""
        from pipeline.helpers.financial.microstructure import (
            analyze_microstructure_patterns,
        )

        if not context:
            raise ValueError("Context required for microstructure detection")

        assert context is not None  # Type guard for linter
        analyzer = context["currentProperties"]["analyzer"]
        catch22_features = {
            k: v for k, v in analyzer.items() if k.startswith("c22_") and not pd.isna(v)
        }

        microstructure_result = analyze_microstructure_patterns(catch22_features)

        if microstructure_result["status"] != "success":
            raise RuntimeError(
                f"Microstructure analysis failed: {microstructure_result.get('message')}"
            )

        result = microstructure_result["result"]
        # Microstructure applicable for intraday only (seconds/minutes/hours, NOT days/weeks/months)
        applicable = context["interval"].endswith(("s", "min", "h"))

        return {
            "hft_noise_level": result["hft_noise_level"],
            "hft_patterns": result["hft_patterns"],
            "applicable_for_data": applicable,
        }

    def _combine_results_enhanced_decision_tree(
        self,
        execution_result: Dict[str, Any],
        data: pd.DataFrame,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Combine results with adaptive weighting and quality-based warnings (v2.0)."""
        method_results = execution_result["method_results"]
        metadata = execution_result["metadata"]
        regime_info = metadata["regime_info"]

        consensus_result = self._build_consensus(method_results, context)
        enhanced_columns = self._create_enhanced_columns(
            data, consensus_result, method_results, context
        )

        result = {
            "outliers": consensus_result["outliers"],
            "outlier_confidence": consensus_result["confidence"],
            **enhanced_columns,
            "outlier_count": int(consensus_result["outliers"].sum()),
        }

        metadata.update(
            {
                "context_reuse_ratio": 0.95 if metadata["tier_reached"] == 1 else 0.92,
                "quality_score": float(consensus_result["confidence"].mean()),
                "adaptive_weights": consensus_result["adaptive_weights"],
                "decomposition_quality_score": consensus_result["quality_score"],
                "residual_strength": consensus_result["residual_strength"],
                "quality_warnings": self._generate_quality_warnings(
                    consensus_result["quality_score"]
                ),
                "skipped_methods": consensus_result["skipped_methods"],
            }
        )

        return {"status": "success", "result": result, "metadata": metadata}

    def _build_consensus(
        self,
        method_results: Dict[str, Dict[str, Any]],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build weighted consensus with quality-based adaptive weighting (v2.0)."""
        if not method_results:
            return {
                "outliers": pd.Series(False, dtype=bool),
                "confidence": pd.Series(0.0, dtype=float),
            }

        index = next(iter(method_results.values()))["outliers"].index

        # Extract quality metrics
        try:
            decomp = context["currentProperties"]["decomposition"]
            quality_score, residual_strength = (
                decomp["quality_score"],
                decomp["residual_strength"],
            )
        except (KeyError, TypeError) as e:
            logging.error(
                f"{self} - Quality metrics extraction failed: {e}. Using fallback weights."
            )
            quality_score, residual_strength = 0.5, 0.02

        adaptive_weights = self._get_adaptive_weights(quality_score, residual_strength)

        # Filter methods by weight
        valid_methods = {
            m: r for m, r in method_results.items() if adaptive_weights[m] > 0.0
        }
        skipped_methods = [
            {"method": m, "reason": f"Zero weight (quality={quality_score:.3f} < 0.5)"}
            for m in method_results
            if adaptive_weights[m] == 0.0
        ]

        if not valid_methods:
            raise ValueError(
                f"All methods filtered by adaptive weighting "
                f"(quality={quality_score:.3f}, skipped={[s['method'] for s in skipped_methods]})"
            )

        # Weighted consensus
        weighted_scores = pd.Series(0.0, index=index, dtype=float)
        for method, result in valid_methods.items():
            weighted_scores += result["consensus_strength"] * adaptive_weights[method]

        total_weight = sum(adaptive_weights[m] for m in valid_methods)
        if total_weight > 0:
            weighted_scores /= total_weight

        return {
            "outliers": weighted_scores > 0.5,
            "confidence": weighted_scores,
            "adaptive_weights": adaptive_weights,
            "skipped_methods": skipped_methods,
            "quality_score": quality_score,
            "residual_strength": residual_strength,
        }

    def _generate_quality_warnings(self, quality_score: float) -> List[str]:
        """Generate quality warnings based on decomposition quality (v2.0)."""
        warnings = []
        if quality_score < 0.5:
            warnings.append(
                f"CRITICAL: Poor decomposition quality ({quality_score:.3f}). "
                "Component detection DISABLED. Using raw price statistics only."
            )
        elif quality_score < 0.7:
            warnings.append(
                f"WARNING: Medium decomposition quality ({quality_score:.3f}). "
                "Using balanced or conservative weighting."
            )
        return warnings

    def _get_adaptive_weights(
        self, quality_score: float, residual_strength: float
    ) -> Dict[str, float]:
        """Quality-based adaptive weighting (v2.0): HIGH≥0.7→30/70, MEDIUM 0.5-0.7→50/50, LOW<0.5→100/0."""
        # CRITICAL: disable components (fail-safe)
        if quality_score < 0.5:
            return {"statistical_enhancement": 1.0, "component_anomaly": 0.0}

        # HIGH quality + STRONG residual: trust components (70%)
        if quality_score >= 0.7 and residual_strength >= 0.05:
            return {"statistical_enhancement": 0.3, "component_anomaly": 0.7}

        # MEDIUM + WEAK residual: conservative (60/40)
        if residual_strength < 0.02:
            return {"statistical_enhancement": 0.6, "component_anomaly": 0.4}

        # Default: balanced (50/50)
        return {"statistical_enhancement": 0.5, "component_anomaly": 0.5}

    def _create_enhanced_columns(
        self,
        data: pd.DataFrame,
        consensus_result: Dict[str, Any],
        method_results: Dict[str, Dict[str, Any]],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, pd.Series]:
        """Create enhanced DataFrame columns."""
        if not context:
            raise ValueError("Context required for enhanced column creation")

        outliers = consensus_result["outliers"]
        confidence = consensus_result["confidence"]
        quality_score = context["currentProperties"]["analyzer"]["data_quality_score"]

        return {
            "outlier_score_enhanced": confidence * quality_score,
            "outlier_type": self._classify_outlier_types(method_results, outliers),
            "price_robust": self._create_robust_price(
                data, outliers, context["targetColumn"]
            ),
        }

    def _classify_outlier_types(
        self, method_results: Dict[str, Dict[str, Any]], consensus_outliers: pd.Series
    ) -> pd.Series:
        """Classify outlier types."""
        outlier_type = pd.Series("none", index=consensus_outliers.index, dtype=str)

        tier1 = method_results.get("statistical_enhancement", {}).get(
            "outliers", pd.Series(False, index=consensus_outliers.index)
        )
        tier2 = method_results.get("component_anomaly", {}).get(
            "outliers", pd.Series(False, index=consensus_outliers.index)
        )

        outlier_type[tier1 & tier2 & consensus_outliers] = "consensus"
        outlier_type[tier1 & ~tier2 & consensus_outliers] = "statistical"
        outlier_type[~tier1 & tier2 & consensus_outliers] = "component"

        return outlier_type

    def _create_robust_price(
        self, data: pd.DataFrame, outliers: pd.Series, target_column: str
    ) -> pd.Series:
        """Create robust price via interpolation."""
        if target_column not in data.columns:
            return pd.Series(0.0, index=outliers.index, dtype=float)

        robust_price = data[target_column].copy()
        robust_price[outliers] = np.nan
        robust_price = robust_price.interpolate(method="linear", limit_direction="both")
        robust_price = robust_price.ffill().bfill()

        return robust_price
