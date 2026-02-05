"""
Processor for detecting periodicity in time series.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from pipeline.helpers.configs import InstrumentTypeConfig
from pipeline.timeSeriesProcessing.baseModule.baseProcessor import BaseProcessor
from pipeline.timeSeriesProcessing.periodicity.algorithmPeriodicityDetector import (
    PeriodicityDetector,
)
from pipeline.timeSeriesProcessing.periodicity.configPeriodicity import (
    RESEARCH_CONFIG,
    build_config_from_properties,
)

__version__ = "1.0.0"


class PeriodicityDetectorProcessor(BaseProcessor):
    """
    Processor for detecting periodicity in time series.

    Manages the lifecycle of periodicity detection: configuration,
    database integration, error handling, and passing results through the pipeline.
    """

    def __init__(
        self,
        ts_id: str,
        currency: str,
        interval: str,
        instrument_type,
        targetColumn: str,
        properties: Optional[Dict[str, Any]] = None,
        detectorConfig: Optional[Dict[str, Any]] = None,
        fallbackBehavior: str = "error",
    ) -> None:
        """
        Initialize periodicity processor.

        Args:
            ts_id: Time series identifier
            currency: Instrument currency
            interval: Data interval
            instrument_type: Instrument type
            targetColumn: Target column for analysis
            properties: Existing properties from database
            detectorConfig: Configuration for periodicity detector (if None - auto-configuration)
            fallbackBehavior: Behavior on errors ('error', 'simple')
        """
        super().__init__(
            ts_id=ts_id,
            currency=currency,
            interval=interval,
            instrument_type=instrument_type,
            targetColumn=targetColumn,
            properties=properties,
            config=detectorConfig,
            fallbackBehavior=fallbackBehavior,
            module_name="periodicity",
        )

    # ========== IMPLEMENTATION OF ABSTRACT METHODS BaseProcessor ==========

    def _execute_algorithm(
        self, series: pd.Series, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute periodicity detection algorithm."""
        try:
            # Get configuration
            if self.config:
                logging.info(
                    f"{self.__str__()} Using custom configuration"
                )
            else:
                self.config = self._build_adaptive_config(series, context)

            # Initialize detector (lazy)
            if self.algorithm is None:
                self.algorithm = self._initialize_algorithm()

            # Execute detection
            detection_result = self.algorithm.process(series, context)

            # Add heuristic periods as suggested_periods
            if detection_result["status"] == "success":
                suggested_periods = self._get_heuristic_periods(self.interval)
                detection_result["result"]["suggested_periods"] = suggested_periods

            return detection_result

        except Exception as e:
            return {
                "status": "error",
                "message": f"Error during detection execution: {str(e)}",
                "metadata": {"error_type": type(e).__name__},
            }

    def _extract_properties(self, algorithm_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract periodicity properties from result.

        UPDATE CONTEXT PERIODICITY - filtering point for what from context goes to DB
        """
        results = algorithm_result["result"]

        # Mapping results to ORM fields
        periodicity_props = {
            # Main results
            "main_period": results["main_period"],
            "periods": (
                self._serialize_list(results["periods"]) if results["periods"] else None
            ),
            "period_confidence_scores": (
                self._serialize_list(results["confidence_scores"])
                if results["confidence_scores"]
                else None
            ),
            # Detection method
            "periodicity_detection_method": results["detection_method"],
            # Detailed method results (memory-optimized)
            "periodicity_method_results": self._serialize_list(
                algorithm_result["metadata"]["method_metadata"]
            ),
            # ACF values (with size limit)
            "acf_values": (
                self._serialize_list(results.get("acf_values", []))
                if results.get("acf_values")
                else None
            ),
            "config_periodicity": self.config,
            # Additional fields
            "suggested_periods": (
                self._serialize_list(results["suggested_periods"])
                if results["suggested_periods"]
                else None
            ),
            "periodicity_quality_score": results["periodicity_quality_score"],
            "detection_status": results["detection_status"],
        }

        # Additional processing for cryptocurrencies
        if self.instrument_type == InstrumentTypeConfig.CRYPTO:
            from pipeline.helpers.configs import PropertySourceConfig

            # Extract analyzer_props from context for crypto adjustments
            analyzer_props = {}  # Fallback if no analyzer data
            self._adjust_for_crypto(periodicity_props, analyzer_props)

        return periodicity_props

    def _initialize_algorithm(self) -> PeriodicityDetector:
        """Initialize periodicity detector algorithm."""
        return PeriodicityDetector(self.config)

    def _validate_properties(self, props: Optional[Dict[str, Any]]) -> bool:
        """Validate received properties."""
        if not props:
            return False

        required_fields = ["main_period"]

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
            "main_period": 0,
            "periods": None,  # NULL for undefined values
            "period_confidence_scores": None,  # NULL for undefined values
            "periodicity_detection_method": "none",
            "periodicity_method_results": json.dumps({}),
            "acf_values": None,  # NULL for undefined values
            "suggested_periods": None,  # NULL for undefined values
            "detection_status": "not_detected",
        }

    def _log_success_summary(self, properties: Dict[str, Any]) -> None:
        """Log successful detection completion."""
        main_period = properties["main_period"]
        detection_status = properties["detection_status"]

        logging.info(
            f"{self.__str__()} Detection completed: "
            f"period={main_period}, "
            f"status={detection_status}"
        )

    def _build_adaptive_config(
        self, series: pd.Series, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build adaptive configuration based on data characteristics."""
        data_length = len(series)

        # Get characteristics from analyzer
        analyzer_props = context["currentProperties"]["analyzer"]

        # Parameters for adaptation
        params = {
            "instrument_type": self.instrument_type,
            "interval": self.interval,
            "data_length": data_length,
            "volatility": analyzer_props["volatility"],
            "stationarity": analyzer_props["is_stationary"],
            "noise_level": analyzer_props["noise_level"],
            "estimated_trend_strength": analyzer_props["estimated_trend_strength"],
        }

        # Log adaptation parameters
        logging.info(
            f"{self.__str__()} Auto-configuration: {self.instrument_type.value}/{self.interval}, "
            f"length={data_length}, volatility={params['volatility']:.2f}, "
            f"stationary={params['stationarity']}"
        )

        try:
            config = build_config_from_properties(params)
            return config
        except ValueError as e:
            logging.warning(
                f"{self.__str__()} Auto-configuration error: {e}, using RESEARCH_CONFIG"
            )
            return RESEARCH_CONFIG

    # ========== OVERRIDE BASEPROCESSOR METHODS ==========

    def _restore_module_state(self) -> None:
        """
        Restore config_periodicity on rerun.

        CRITICAL FIX: Restore self.config from properties["config_periodicity"]
        for correct algorithm operation on rerun.
        """
        # Restore config from properties
        if self.properties and "config_periodicity" in self.properties:
            self.config = self.properties["config_periodicity"]
            logging.info(
                f"{self.__str__()} Rerun: restored config_periodicity from DB"
            )
        else:
            logging.warning(
                f"{self.__str__()} Rerun: config_periodicity not found in properties, "
                "will create adaptive config"
            )

    # ========== ADDITIONAL METHODS ==========

    def _serialize_list(self, lst: List) -> str:
        """Safe list serialization to JSON."""
        try:
            return json.dumps(lst)
        except (TypeError, ValueError) as e:
            logging.warning(f"{self.__str__()} List serialization error: {e}")
            return json.dumps([])

    def _adjust_for_crypto(
        self, periodicity_props: Dict[str, Any], analyzer_props: Dict[str, Any]
    ) -> None:
        """Special adjustment for cryptocurrencies."""
        # If very long period found for cryptocurrencies
        main_period = periodicity_props["main_period"]
        if main_period > 500:  # Suspiciously long period
            logging.info(
                f"{self.__str__()} Detected long period {main_period} for cryptocurrency, "
                f"checking reliability"
            )

            # Reduce confidence if volatility is high
            if analyzer_props and "volatility" in analyzer_props:
                volatility = analyzer_props["volatility"]
                if volatility > 0.7:
                    try:
                        scores_str = periodicity_props.get("period_confidence_scores")
                        if scores_str:
                            scores = json.loads(scores_str)
                            if scores:
                                scores[
                                    0
                                ] *= 0.7  # Reduce confidence of main period
                                periodicity_props["period_confidence_scores"] = (
                                    json.dumps(scores)
                                )
                                logging.debug(
                                    f"{self.__str__()} Reduced confidence due to high volatility"
                                )
                    except Exception as e:
                        logging.warning(
                            f"{self.__str__()} Scores adjustment error: {e}"
                        )

    def _get_heuristic_periods(self, interval: str) -> List[int]:
        """Get heuristic periods for interval."""
        # Heuristic periods table for cryptocurrencies
        heuristic_map = {
            # Seconds
            "1s": [60, 3600],  # Minute, hour
            "15s": [4, 240],  # Minute, hour
            "30s": [2, 120],  # Minute, hour
            # Minutes
            "1m": [60, 1440],  # Hour, day
            "3m": [20, 480],  # Hour, day
            "5m": [12, 288],  # Hour, day
            "15m": [4, 96],  # Hour, day
            "30m": [2, 48],  # Hour, day
            # Hours
            "1h": [24, 168],  # Day, week
            "3h": [8, 56],  # Day, week
            "6h": [4, 28],  # Day, week
            "12h": [2, 14],  # Day, week
            # Days
            "1d": [7, 30],  # Week, month
            "3d": [2, 10],  # Week, month
            "1w": [4, 52],  # Month, year
            "1M": [12],  # Year
        }

        return heuristic_map.get(interval, [])

    # ========== HEURISTIC FALLBACK IMPLEMENTATION ==========

    def _get_heuristic_fallback_values(self) -> Dict[str, Any]:
        """
        Heuristic fallback values for PeriodicityDetectorProcessor.

        FIX SB8-68: Restores legacy behavior of adding
        heuristic periods on processing errors.

        Implements logic from legacy processorPeriodicityDetector__.py lines 362-369:
        - suggested_periods from _get_heuristic_periods(self.interval)
        - main_period as first element of suggested_periods

        Returns:
            Dict with heuristic fallback values:
            - suggested_periods: JSON string with heuristic periods
            - main_period: int, first period from heuristics (fallback main period)
        """
        try:
            # Get heuristic periods (legacy logic)
            suggested_periods = self._get_heuristic_periods(self.interval)

            heuristic_values = {}

            # Add suggested_periods (as in legacy code)
            if suggested_periods:
                heuristic_values["suggested_periods"] = json.dumps(suggested_periods)
                heuristic_values["main_period"] = suggested_periods[
                    0
                ]  # First period as main

                logging.debug(
                    f"{self.__str__()} Heuristic fallback: suggested_periods={suggested_periods}, "
                    f"main_period={suggested_periods[0]}"
                )
            else:
                # Fallback to universal values if interval unknown
                heuristic_values["suggested_periods"] = json.dumps([])
                heuristic_values["main_period"] = 0

                logging.debug(
                    f"{self.__str__()} No heuristic periods for interval '{self.interval}', "
                    "using empty fallback"
                )

            return heuristic_values

        except Exception as e:
            logging.warning(
                f"{self.__str__()} Error generating heuristic fallback values: {e}"
            )
            return {}