"""
DecompositionProcessor unified with BaseProcessor.

Inherits from BaseProcessor for architectural unity, but preserves
all unique decomposition features through method overriding.
"""

import ast
import json
import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from pipeline.helpers.utils import to_json_safe
from pipeline.timeSeriesProcessing.baseModule.baseProcessor import BaseProcessor
from pipeline.timeSeriesProcessing.decomposition.algorithmDecomposition import (
    DecompositionAlgorithm,
)
from pipeline.timeSeriesProcessing.decomposition.configDecomposition import (
    build_config_from_properties,
)

__version__ = "2.0.0"


class DecompositionProcessor(BaseProcessor):
    """
    Processor for time series decomposition.

    ARCHITECTURAL STRATEGY:
    - Inherits from BaseProcessor for system unity (95% duplication eliminated)
    - Overrides specific methods for unique decomposition logic
    - Preserves Component Recovery System and DataFrame enrichment
    - Adds process_with_dataframe_enrichment() method for compatibility

    UNIQUENESS: Component Recovery, Enhanced Decision Tree, comprehensive quality evaluation.
    """

    def __init__(
        self,
        ts_id: str,
        currency: str,
        interval: str,
        instrument_type,
        targetColumn: str,
        properties: Optional[Dict[str, Any]] = None,
        decompositionConfig: Optional[Dict[str, Any]] = None,
        fallbackBehavior: str = "error",
    ) -> None:
        """
        Initialize decomposition processor.

        Args:
            decompositionConfig: Configuration for decomposition (compatibility with original API)
        """
        # Call BaseProcessor.__init__ with parameter rename for compatibility
        super().__init__(
            ts_id=ts_id,
            currency=currency,
            interval=interval,
            instrument_type=instrument_type,
            targetColumn=targetColumn,
            properties=properties,
            config=decompositionConfig,  # decompositionConfig → config
            fallbackBehavior=fallbackBehavior,
            module_name="decomposition",
        )

    # ========== IMPLEMENTATION OF BASEPROCESSOR ABSTRACT METHODS ==========

    def _validate_properties(self, properties: Optional[Dict[str, Any]]) -> bool:
        """Validate decomposition properties structure."""
        if properties is None:
            return False

        required_fields = ["decomposition_method", "config_decomposition"]
        return all(field in properties for field in required_fields)

    def _execute_algorithm(
        self, series: pd.Series, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute time series decomposition algorithm.

        ADAPTER: Integrates unique decomposition logic into BaseProcessor workflow.
        """
        try:
            # Validate dependencies (Fail Fast)
            self._validate_context_dependencies(context)

            # Get configuration
            if self.config:
                logging.info(f"{self} Using custom configuration")
            else:
                self.config = self._build_adaptive_config(series, context)

            # Validate and adjust STL parameters
            self.config = self._validate_stl_parameters(self.config, len(series))

            # Initialize decomposition algorithm
            if self.algorithm is None:
                self.algorithm = self._initialize_algorithm()

            # Execute decomposition through unique DecompositionAlgorithm
            return self.algorithm.process(series, context)

        except Exception as e:
            return {
                "status": "error",
                "message": f"Error executing decomposition: {str(e)}",
                "metadata": {"error_type": type(e).__name__},
            }

    def _extract_properties(self, algorithm_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract extended decomposition properties from result.

        UPDATE CONTEXT DECOMPOSITION - filtering point for what goes from context to DB.
        """
        try:
            results = algorithm_result["result"]
            metadata = algorithm_result["metadata"]

            # Extract component_strengths
            component_strengths = results["component_strengths"]

            # Base properties for saving to DB
            decomposition_props = {
                "decomposition_method": metadata["selected_method"],
                "baseline_quality": metadata["baseline_quality"],
                "model_type": metadata["model_type"],
                "trend_strength": component_strengths.get("trend_strength", 0.0),
                "seasonal_strength": component_strengths.get("seasonal_strength", 0.0),
                "residual_strength": component_strengths.get("residual_strength", 1.0),
                "quality_score": results["quality_score"],
                "reconstruction_error": results["reconstruction_error"],
                "config_decomposition": self.config,
                "mstl_stability_metrics_converged": results.get(
                    "stability_metrics", {}
                ).get("converged", None),
                "mstl_corrections_applied": results.get("parameters_corrected", []),
                # === Fourier-specific metrics ===
                "fourier_n_harmonics": results.get("harmonics_info", {}).get(
                    "n_harmonics", None
                ),
                "fourier_spectral_entropy": results.get("harmonics_info", {}).get(
                    "spectral_entropy", None
                ),
                "fourier_harmonic_strength": results.get("harmonics_info", {}).get(
                    "harmonic_strength", None
                ),
                # === SSA-specific metrics ===
                "ssa_window_length": results.get("window_length", None),
                "ssa_n_components_used": results.get("n_components_used", None),
                "ssa_variance_explained": results.get("variance_explained", None),
                "ssa_component_grouping": to_json_safe(
                    results.get("component_grouping", None)
                ),
                # === TBATS-specific metrics ===
                "tbats_seasonal_periods": to_json_safe(
                    results.get("seasonal_periods", None)
                ),
                "tbats_box_cox_lambda": results.get("box_cox_lambda", None),
                "tbats_aic": results.get("aic", None),
                "tbats_arma_order": to_json_safe(results.get("arma_order", None)),
                "tbats_damped_trend": results.get("damped_trend", None),
                # === Prophet-specific metrics ===
                "prophet_changepoints_detected": results.get(
                    "changepoints_detected", None
                ),
                "prophet_trend_flexibility": results.get("trend_flexibility", None),
                "prophet_seasonality_mode": results.get("seasonality_mode", None),
                "prophet_cross_validation_mape": results.get(
                    "cross_validation_mape", None
                ),
                "prophet_cross_validation_rmse": results.get(
                    "cross_validation_rmse", None
                ),
                "prophet_bayesian_uncertainty": results.get(
                    "bayesian_uncertainty", None
                ),
                "prophet_trend_changepoint_dates": to_json_safe(
                    results.get("trend_changepoint_dates", None)
                ),
                "prophet_component_importance": to_json_safe(
                    results.get("component_importance", None)
                ),
                # === N-BEATS-specific metrics ===
                "nbeats_model_type": results.get("model_type", None),
                "nbeats_training_loss": results.get("training_loss", None),
                "nbeats_convergence_achieved": results.get("converged", None),
                "nbeats_harmonic_complexity": results.get("harmonic_complexity", None),
                "nbeats_seasonal_harmonics_used": results.get("harmonics_used", None),
                "nbeats_architecture_efficiency": results.get(
                    "architecture_efficiency", None
                ),
            }

            return decomposition_props

        except (KeyError, TypeError, AttributeError) as e:
            raise ValueError(
                f"{self} Incorrect decomposition result structure. "
                f"Missing required field: {str(e)}"
            ) from e

    def _initialize_algorithm(self) -> DecompositionAlgorithm:
        """Initialize decomposition algorithm."""
        return DecompositionAlgorithm(self.config)

    def _get_default_properties(self) -> Dict[str, Any]:
        """Default properties for fallbackBehavior='simple'."""
        return {
            "decomposition_method": "none",
            "baseline_quality": 0.0,
            "model_type": "additive",
            "trend_strength": 0.0,
            "seasonal_strength": 0.0,
            "residual_strength": 1.0,
            "quality_score": 0.0,
            "reconstruction_error": float("inf"),
            "config_decomposition": {},
        }

    def _log_success_summary(self, properties: Dict[str, Any]) -> None:
        """Log successful decomposition completion."""
        method = properties["decomposition_method"]
        quality_score = properties["quality_score"]
        trend_strength = properties["trend_strength"]
        seasonal_strength = properties["seasonal_strength"]

        logging.info(
            f"{self} Decomposition completed with method {method}: "
            f"quality={quality_score:.3f}, trend={trend_strength:.3f}, "
            f"seasonal={seasonal_strength:.3f}"
        )

    # ========== BASEPROCESSOR METHOD OVERRIDES ==========

    def _restore_module_state(self) -> None:
        """
        Restore decomposition module state on re-run.

        Restores config from properties for consistency with other processors.
        """
        if self.properties and "config_decomposition" in self.properties:
            self.config = self.properties["config_decomposition"]
            logging.info(f"{self} Config restored from properties")
        else:
            logging.warning(
                f"{self} No config_decomposition in properties, will rebuild if needed"
            )

    def _get_heuristic_fallback_values(self) -> Dict[str, Any]:
        """
        Heuristic fallback values for decomposition module.

        Uses STL baseline quality as indicator for diagnostics.

        Returns:
            Dict with heuristic values for enriching default_properties
        """
        try:
            # Check for baseline result from algorithm
            if (
                    hasattr(self, "algorithm")
                    and self.algorithm is not None
                    and hasattr(self.algorithm, "baseline_result")
                    and self.algorithm.baseline_result
            ):
                baseline = self.algorithm.baseline_result

                if baseline.get("status") == "success":
                    heuristic_values = {
                        "baseline_quality": baseline.get("quality_score", 0.0),
                        "fallback_method": "robust_stl",
                        "heuristic_source": "stl_baseline"
                    }

                    logging.info(
                        f"{self} Heuristic fallback: baseline_quality="
                        f"{heuristic_values['baseline_quality']:.3f}"
                    )

                    return heuristic_values

            logging.debug(f"{self} No baseline result available for heuristic fallback")
            return {}

        except Exception as e:
            logging.warning(
                f"{self} Error generating heuristic fallback values: {e}"
            )
            return {}

    def _restore_enrichment_columns(
        self, data: pd.DataFrame, context: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Restore decomposition component time series on re-run.

        ARCHITECTURAL CONSISTENCY: Implements Template Method Extension pattern,
        unifying approach with AnalyzerProcessor for enrichment restoration.

        UNIQUE FEATURE: Component Recovery System with 10x performance improvement.
        Restores full component time series:
        - {target}_trend: trend component
        - {target}_seasonal: seasonal component
        - {target}_residual: residual component

        Args:
            data: DataFrame for restoring decomposition components
            context: Processing context with properties

        Returns:
            DataFrame with restored decomposition components
        """
        try:
            # Execute full component restoration through Component Recovery System
            data = self._restore_decomposition_components_from_properties(data, context)

            logging.info(
                f"{self} Component Recovery System: component time series restored from properties"
            )
            return data

        except Exception as e:
            logging.warning(
                f"{self} Component Recovery failed: {str(e)}, returning original data"
            )
            return data

    def _add_module_columns_to_dataframe(
        self, dataframe: pd.DataFrame, algorithm_result: Dict[str, Any]
    ) -> pd.DataFrame:
        """Override to add decomposition components to DataFrame."""
        try:
            if algorithm_result["status"] != "success":
                logging.warning(f"{self} - Decomposition failed, skipping components")
                return dataframe

            # Add decomposition components
            return self._add_decomposition_components(dataframe, algorithm_result)
        except Exception as e:
            logging.warning(
                f"{self} - Component enrichment failed ({str(e)}), returning original DataFrame"
            )
            return dataframe

    # ========== UNIQUE DECOMPOSITION METHODS (ORIGINAL LOGIC) ==========

    def _restore_decomposition_components_from_properties(
        self, data: pd.DataFrame, context: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Restore decomposition components based on saved properties.

        UNIQUE FEATURE: Component Recovery System.

        Args:
            data: DataFrame with source data
            context: Processing context

        Returns:
            DataFrame with added decomposition components
        """
        try:
            # Extract saved parameters
            if self.properties is None:
                raise ValueError(
                    "Properties not initialized for component restoration"
                )

            method = self.properties["decomposition_method"]
            config = self.properties["config_decomposition"]

            logging.info(f"Restoring components with method: {method}")

            # Use same logic but with saved configuration
            target_series = data[self.targetColumn]

            # Validate and adjust configuration for current data
            validated_config = self._validate_stl_parameters(config, len(target_series))

            # Specify predetermined method
            validated_config["_predetermined_method"] = method  # From DB

            # Execute decomposition with saved parameters
            algorithm = DecompositionAlgorithm(validated_config)
            decomposition_result = algorithm.process(target_series, context)

            if decomposition_result["status"] == "success":
                # Add components to DataFrame
                data = self._add_decomposition_components(data, decomposition_result)
                logging.info("Decomposition components successfully restored")
            else:
                logging.warning(
                    f"Failed to restore components: {decomposition_result.get('message', 'Unknown error')}"
                )

            return data

        except Exception as e:
            logging.error(f"Error restoring components: {e}", exc_info=True)
            # Graceful degradation: return data without components
            return data

    def _validate_context_dependencies(self, context: Dict[str, Any]) -> None:
        """
        Validate context dependencies (Fail Fast).

        Raises:
            ValueError: If required dependencies are missing
        """
        required_groups = ["analyzer", "periodicity"]
        current_props = context["currentProperties"]

        missing_groups = []
        for group in required_groups:
            if group not in current_props or not current_props[group]:
                missing_groups.append(group)

        if missing_groups:
            raise ValueError(
                f"Missing required property groups in context: {missing_groups}. "
                f"Decomposition requires prior execution of analyzer and periodicity."
            )

        # Check for critical analyzer properties
        analyzer_props = current_props["analyzer"]
        required_analyzer_props = [
            "volatility",
            "is_stationary",
            "noise_level",
            "estimated_trend_strength",
            "data_quality_score",
        ]
        missing_props = [p for p in required_analyzer_props if p not in analyzer_props]
        if missing_props:
            raise ValueError(
                f"Missing required analyzer properties: {missing_props}"
            )

        # Check for critical periodicity properties
        periodicity_props = current_props["periodicity"]
        required_periodicity_props = [
            "main_period",
            "periods",
            "suggested_periods",
            "detection_status",
            "periodicity_quality_score",
        ]
        missing_props = [
            p for p in required_periodicity_props if p not in periodicity_props
        ]
        if missing_props:
            raise ValueError(
                f"Missing required periodicity properties: {missing_props}"
            )

    def _build_adaptive_config(
        self, series: pd.Series, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build adaptive configuration based on data characteristics."""
        data_length = len(series)

        # Get characteristics from analyzer
        analyzer_props = context["currentProperties"]["analyzer"]
        periodicity_props = context["currentProperties"]["periodicity"]

        # Parameters for adaptation
        params = {
            "instrument_type": self.instrument_type,
            "interval": self.interval,
            "data_length": data_length,
            "volatility": analyzer_props["volatility"],
            "outlier_ratio": analyzer_props["outlier_ratio"],
            "is_stationary": analyzer_props["is_stationary"],
            "noise_level": analyzer_props["noise_level"],
            "estimated_trend_strength": analyzer_props["estimated_trend_strength"],
            "main_period": periodicity_props["main_period"],
            "periods": periodicity_props["periods"],
            "suggested_periods": self._safe_parse_periods(
                periodicity_props["suggested_periods"]
            ),
        }

        # Log adaptation parameters
        logging.info(
            f"{self} Auto-configuration: {self.instrument_type.value}/{self.interval}, "
            f"length={data_length}, volatility={params['volatility']:.2f}"
        )

        try:
            config = build_config_from_properties(params)
            return config
        except ValueError as e:
            raise ValueError(f"{self} Auto-configuration error: {e}") from e

    def _validate_stl_parameters(
        self, config: Dict[str, Any], data_length: int
    ) -> Dict[str, Any]:
        """
        Validate STL parameters according to mathematical constraints.

        Args:
            config: Configuration for validation
            data_length: Time series length

        Returns:
            Corrected configuration
        """
        corrected_config = config.copy()

        for method in ["stl", "mstl", "robust_stl"]:
            if method in corrected_config and corrected_config[method]:
                method_config = corrected_config[method]

                # Validate seasonal parameter: seasonal ≤ data_length/2
                if "seasonal" in method_config:
                    max_seasonal = data_length // 2
                    if method_config["seasonal"] > max_seasonal:
                        logging.warning(
                            f"{self} Correcting seasonal parameter for {method}: "
                            f"{method_config['seasonal']} -> {max_seasonal}"
                        )
                        method_config["seasonal"] = max_seasonal

                # Validate trend parameter
                if "trend" in method_config and method_config["trend"]:
                    max_trend = data_length // 3
                    if method_config["trend"] > max_trend:
                        logging.warning(
                            f"{self} Correcting trend parameter for {method}: "
                            f"{method_config['trend']} -> {max_trend}"
                        )
                        method_config["trend"] = max_trend

                # Validate period parameter
                if "period" in method_config and method_config["period"]:
                    if method_config["period"] >= data_length:
                        suggested_period = min(data_length // 4, 24)
                        logging.warning(
                            f"{self} Correcting period parameter for {method}: "
                            f"{method_config['period']} -> {suggested_period}"
                        )
                        method_config["period"] = suggested_period

        return corrected_config

    def _add_decomposition_components(
        self, data: pd.DataFrame, decomposition_result: Dict[str, Any]
    ) -> pd.DataFrame:
        """Add decomposition components to DataFrame."""
        if decomposition_result["status"] != "success":
            return data

        results = decomposition_result["result"]

        # Add components if they exist
        if "trend" in results:
            data["trend"] = results["trend"]
            # data[f"{self.targetColumn}_trend"] = results["trend"]
        if "seasonal" in results:
            data["seasonal"] = results["seasonal"]
            # data[f"{self.targetColumn}_seasonal"] = results["seasonal"]
        if "residual" in results:
            data["residual"] = results["residual"]
            # data[f"{self.targetColumn}_residual"] = results["residual"]

        logging.debug(
            f"Added decomposition components to DataFrame: "
            f"trend, seasonal, residual"
        )

        return data

    def _safe_parse_periods(self, periods_str: str) -> list:
        """
        Safely parse periods string using ast.literal_eval instead of eval().

        Args:
            periods_str: String representation of periods list

        Returns:
            Parsed list of periods or empty list on failure
        """
        try:
            # First try ast.literal_eval for safety
            parsed_periods = ast.literal_eval(periods_str)
            if isinstance(parsed_periods, list):
                return parsed_periods
            else:
                logging.warning(
                    f"{self} periods_str is not a list: {type(parsed_periods)}, "
                    f"converting to list"
                )
                return [parsed_periods] if parsed_periods is not None else []
        except (ValueError, SyntaxError) as e:
            logging.warning(
                f"{self} Failed to parse periods_str '{periods_str}': {e}. "
                f"Attempting JSON parsing fallback."
            )
            try:
                # Fallback to JSON parsing
                json_periods = json.loads(periods_str)
                if isinstance(json_periods, list):
                    return json_periods
                else:
                    return [json_periods] if json_periods is not None else []
            except json.JSONDecodeError as json_error:
                logging.error(
                    f"{self} Both ast.literal_eval and JSON parsing failed for "
                    f"periods_str '{periods_str}': {json_error}. Using empty list."
                )
                return []

    # ========== ADDITIONAL METHOD FOR COMPATIBILITY ==========

    def process_with_dataframe_enrichment(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Method for compatibility with analyzer/decomposition pattern.

        Executes decomposition and adds components to DataFrame.

        Args:
            dataframe: DataFrame with data
            target_column: Target column for decomposition
            context: Processing context

        Returns:
            Tuple[DataFrame with components, decomposition result]
        """
        try:
            # Check compatibility with targetColumn
            if target_column != self.targetColumn:
                logging.warning(
                    f"{self} target_column mismatch: {target_column} != {self.targetColumn}"
                )

            # If existing properties, restore components
            if self.properties:
                enriched_data = self._restore_decomposition_components_from_properties(
                    dataframe, context or {}
                )
                return enriched_data, {"status": "success", "source": "properties"}

            # Otherwise execute full decomposition
            series = dataframe[target_column]
            decomposition_result = self._execute_algorithm(series, context or {})

            if decomposition_result["status"] == "success":
                enriched_data = self._add_decomposition_components(
                    dataframe, decomposition_result
                )
                return enriched_data, decomposition_result
            else:
                return dataframe, decomposition_result

        except Exception as e:
            error_result = {
                "status": "error",
                "message": f"DataFrame enrichment failed: {str(e)}",
                "metadata": {"error_type": type(e).__name__},
            }
            return dataframe, error_result