"""
Time series analysis processor - pipeline integration.
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from pipeline.timeSeriesProcessing.analyzer.algorithmAnalyzer import TimeSeriesAnalyzer
from pipeline.timeSeriesProcessing.analyzer.configAnalyzer import (
    build_config_from_properties,
)
from pipeline.timeSeriesProcessing.baseModule.baseProcessor import BaseProcessor

__version__ = "1.0.0"


class AnalysisProcessor(BaseProcessor):
    """
    Processor for time series analysis with support for different instrument types.

    Manages analysis lifecycle: database integration,
    error handling, and passing results through the pipeline.
    """

    def __init__(
        self,
        ts_id: str,
        currency: str,
        interval: str,
        instrument_type,
        targetColumn: str,
        properties: Optional[Dict[str, Any]] = None,
        analyzerConfig: Optional[Dict[str, Any]] = None,
        fallbackBehavior: str = "error",
    ) -> None:
        """
        Initialize analysis processor.

        Args:
            ts_id: Time series identifier
            currency: Instrument currency
            interval: Data interval
            instrument_type: Instrument type
            targetColumn: Target column for analysis
            properties: Existing properties from database
            analyzerConfig: Explicit analyzer configuration (overrides automatic)
            fallbackBehavior: Error behavior ('simple', 'error')
                - 'simple': use simplified analysis
                - 'error': return error (default)
        """
        super().__init__(
            ts_id=ts_id,
            currency=currency,
            interval=interval,
            instrument_type=instrument_type,
            targetColumn=targetColumn,
            properties=properties,
            config=analyzerConfig,
            fallbackBehavior=fallbackBehavior,
            module_name="analyzer",
        )

    # ========== IMPLEMENTATION OF BaseProcessor ABSTRACT METHODS ==========

    def _execute_algorithm(
        self, series: pd.Series, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute time series analysis algorithm."""
        try:
            # Get configuration
            if self.config:
                logging.info(
                    f"{self.__str__()} Using custom configuration"
                )
            else:
                self.config = self._build_adaptive_config(series)

            # Configure return_indices for DataFrame enrichment (not for database storage!)
            if "outlier" in self.config:
                self.config["outlier"]["return_indices"] = True
            else:
                self.config["outlier"] = {"return_indices": True}

            # Initialize analyzer algorithm
            if self.algorithm is None:
                self.algorithm = self._initialize_algorithm()

            # Execute analysis
            return self.algorithm.process(series, context)

        except Exception as e:
            return {
                "status": "error",
                "message": f"Error during analysis execution: {str(e)}",
                "metadata": {"error_type": type(e).__name__},
            }

    def _extract_properties(self, algorithm_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract analyzer properties from result.

        CONTEXT ANALYZER UPDATE - filtering point for what from context goes to database
        """
        if "result" not in algorithm_result:
            error_msg = f"{self.__str__()} No analysis result"
            logging.error(error_msg)
            raise Exception(error_msg)

        results = algorithm_result["result"]

        # Mapping results to expected ORM fields
        analyzer_props = {
            # Basic characteristics
            "length": results["length"],
            "missing_ratio": results["missing_ratio"],
            "missing_values": results["missing_values"],
            # Statistical characteristics
            "volatility": results["volatility"],
            "skewness": results["skewness"],
            "kurtosis": results["kurtosis"],
            "lag1_autocorrelation": results["lag1_autocorrelation"],
            # Outliers (only statistics - not saving indices!)
            "zscore_outliers": results["zscore_outliers"],
            "iqr_outliers": results["iqr_outliers"],
            "mad_outliers": results["mad_outliers"],
            "outlier_ratio": results["outlier_ratio"],
            # Stationarity
            "is_stationary": results["is_stationary"],
            "adf_pvalue": results["adf_pvalue"],
            "kpss_pvalue": results["kpss_pvalue"],
            # Rolling characteristics
            "rolling_mean_cv": results["rolling_mean_cv"],
            "rolling_std_cv": results["rolling_std_cv"],
            # Trend and noise
            "estimated_trend_strength": results["estimated_trend_strength"],
            "noise_level": results["noise_level"],
            # Q-Q statistics
            "qq_correlation": results["qq_correlation"],
            # Additional metrics
            "data_quality_score": results["data_quality_score"],
            "series_type": results.get("series_type"),
            "config_analyzer": self.config,
        }

        # Add catch22 features (if present in results)
        catch22_fields = [
            "c22_mode_5",
            "c22_mode_10",
            "c22_outlier_timing_pos",
            "c22_outlier_timing_neg",
            "c22_acf_timescale",
            "c22_acf_first_min",
            "c22_low_freq_power",
            "c22_centroid_freq",
            "c22_ami_timescale",
            "c22_periodicity",
            "c22_ami2",
            "c22_trev",
            "c22_stretch_high",
            "c22_stretch_decreasing",
            "c22_entropy_pairs",
            "c22_transition_variance",
            "c22_whiten_timescale",
            "c22_high_fluctuation",
            "c22_forecast_error",
            "c22_rs_range",
            "c22_dfa",
            "c22_embedding_dist",
        ]

        for field in catch22_fields:
            if field in results:
                analyzer_props[field] = results[field]

        # Remove None values
        analyzer_props = {k: v for k, v in analyzer_props.items() if v is not None}
        return analyzer_props

    def _initialize_algorithm(self) -> TimeSeriesAnalyzer:
        """Initialize analyzer algorithm."""
        return TimeSeriesAnalyzer(self.config)

    def _validate_properties(self, props: Optional[Dict[str, Any]]) -> bool:
        """Validate obtained properties."""
        if not props:
            return False

        required_fields = [
            "volatility",
            "estimated_trend_strength",
            "is_stationary",
            "outlier_ratio",
        ]

        for field in required_fields:
            if field not in props:
                logging.warning(
                    f"{self.__str__()} Missing required field '{field}' in properties"
                )
                return False

        return True

    def _get_default_properties(self) -> Dict[str, Any]:
        """Default properties for fallbackBehavior='simple'."""
        return {
            "length": 0,
            "missing_ratio": 0.0,
            "missing_values": 0,
            "volatility": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "is_stationary": 0,
            "outlier_ratio": 0.0,
            "estimated_trend_strength": 0.0,
            "noise_level": 0.0,
            "series_type": "simple_fallback",
        }

    def _log_success_summary(self, properties: Dict[str, Any]) -> None:
        """Log successful analysis completion."""
        logging.info(
            f"{self.__str__()} Analysis completed for {self.instrument_type.value}: "
            f"length={properties['length']}, "
            f"volatility={properties['volatility']:.3f}, "
            f"trend={properties['estimated_trend_strength']:.3f}, "
            f"stationary={properties['is_stationary']}, "
            f"outliers={properties['outlier_ratio']:.2%}"
        )

    # ========== OVERRIDING BaseProcessor METHODS ==========

    def _restore_module_state(self) -> None:
        """
        Restore config_analyzer on repeated run.
        Restore self.config from properties["config_analyzer"]
        for correct algorithm operation on repeated run.
        """
        # Restore config from properties
        if self.properties and "config_analyzer" in self.properties:
            self.config = self.properties["config_analyzer"]
            # Initialize algorithm with restored config
            try:
                self.algorithm = self._initialize_algorithm()
                logging.info(
                    f"{self.__str__()} Repeated run: restored config_analyzer + "
                    f"initialized algorithm with {len(self.config.get('_active_methods', []))} methods"
                )
            except Exception as e:
                # If initialization failed, log but don't crash
                # This allows enrichment to attempt algorithm creation later
                logging.error(
                    f"{self.__str__()} Algorithm initialization failed: {e}",
                    exc_info=True
                )
                self.algorithm = None
        else:
            logging.warning(
                f"{self.__str__()} Repeated run: config_analyzer not found in properties, "
                "adaptive config will be created"
            )

    def _restore_enrichment_columns(
            self, data: pd.DataFrame, context: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Recalculate outlier boolean columns for repeated runs.

        Relies on properly initialized self.algorithm from
        _restore_module_state(), avoiding re-initialization.

        CONCEPT: On repeated run, statistics (properties) are reused from database,
        but DataFrame enrichment (outlier columns) is RECALCULATED fresh.

        This ensures:
        - 10x performance boost (don't recalculate statistics)
        - Consistent output format (outlier columns always present)
        - Using exact configuration from database

        Args:
            data: DataFrame to add outlier columns to
            context: Processing context with properties

        Returns:
            DataFrame with recalculated outlier boolean columns
        """
        try:
            # Check that algorithm is initialized
            if self.algorithm is None:
                logging.warning(
                    f"{self.__str__()} Algorithm not initialized in _restore_module_state(), "
                    "attempting lazy initialization"
                )
                # Fallback: use restored config or create adaptive
                if self.config is None:
                    self.config = self._build_adaptive_config(data[self.targetColumn])
                self.algorithm = self._initialize_algorithm()

            logging.info(
                f"{self.__str__()} Recalculating outlier columns with restored config"
            )

            # Use standard DataFrame enrichment logic
            # Algorithm already initialized, enrichment will be efficient
            return self._add_module_columns_to_dataframe(data, {"status": "success"})

        except Exception as e:
            # Graceful degradation - enrichment failure doesn't break pipeline
            logging.warning(
                f"{self.__str__()} Outlier enrichment failed: {str(e)}, "
                "returning original DataFrame without enrichment columns",
                exc_info=True
            )
            return data

    # ========== ADDITIONAL METHODS ==========

    def _build_adaptive_config(self, series: pd.Series) -> Dict[str, Any]:
        """Get configuration specific to instrument type."""
        config = build_config_from_properties(
            params={
                "instrument_type": self.instrument_type,
                "interval": self.interval,
                "data_length": len(series),
            }
        )

        logging.info(
            f"{self.__str__()} Using configuration for {self.instrument_type.value} "
            f"with interval {self.interval}"
        )

        return config

    def _add_module_columns_to_dataframe(
        self, dataframe: pd.DataFrame, algorithm_result: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Override to add outlier boolean columns.

        Args:
            dataframe: Original DataFrame
            algorithm_result: Analysis result from TimeSeriesAnalyzer

        Returns:
            DataFrame with added outlier columns
        """
        try:
            # If analysis failed, return original DataFrame
            if algorithm_result["status"] != "success":
                logging.warning(
                    f"{self} - Analysis failed, skipping outlier enrichment"
                )
                return dataframe

            # Create analyzer configuration
            analyzer_config = self.config or self._build_adaptive_config(
                dataframe[self.targetColumn]
            )

            # Initialize TimeSeriesAnalyzer
            if self.algorithm is None:
                self.algorithm = TimeSeriesAnalyzer(analyzer_config)

            # Execute DataFrame enrichment through Algorithm level
            enriched_df, _ = self.algorithm.process_with_dataframe_enrichment(
                dataframe, self.targetColumn, context=None
            )

            logging.debug(
                f"{self} - Added outlier columns to DataFrame: "
                f"is_zscore_outlier, is_iqr_outlier, is_mad_outlier"
            )

            return enriched_df

        except Exception as e:
            # Fallback - return original DataFrame if enrichment failed
            logging.warning(
                f"{self} - Outlier enrichment failed ({str(e)}), returning original DataFrame"
            )
            return dataframe