import logging
from datetime import datetime
from typing import Dict, List, Union

import pandas as pd

from pipeline.helpers.configs import InstrumentTypeConfig, PropertySourceConfig
from pipeline.helpers.utils import validate_required_locals
from pipeline.timeSeriesProcessing.analyzer.processorAnalyzer import AnalysisProcessor
from pipeline.timeSeriesProcessing.decomposition.processorDecomposition import (
    DecompositionProcessor,
)
from pipeline.timeSeriesProcessing.outlierDetection.processorOutlierDetection import (
    OutlierDetectionProcessor,
)
from pipeline.timeSeriesProcessing.periodicity.processorPeriodicityDetector import (
    PeriodicityDetectorProcessor,
)
from pipeline.timeSeriesProcessing.pipeline import ProcessingPipeline
from pipeline.timeSeriesProcessing.preprocessingConfig import (
    TimeSeriesPreprocessingConfig,
)
from pipeline.timeSeriesProcessing.propertyManager import PropertyManager


class TimeSeriesPreprocessor:
    """Facade for time series preprocessing"""

    def __init__(
        self,
        ts_id: str,
        db_session=None,
        timestampColumn: str = "Open_time",
        targetColumn: str = "Open",
        forceRecalculate: Union[bool, Dict[str, bool]] = False,
        createNewVersion: bool = False
    ):
        """
        Initialize preprocessor

        Args:
            ts_id: Time series identifier
            db_session: Database session
            timestampColumn: Timestamp column name
            targetColumn: Target column name
            forceRecalculate: Force property recalculation
            createNewVersion: Create new property version
        """
        # Validation in one line via locals()
        validate_required_locals(["ts_id", "db_session"], locals())

        # Parse ts_id
        parts = ts_id.split("_")
        if len(parts) < 3:
            raise ValueError(
                f"Invalid ts_id format: {ts_id}. Expected: type_currency_interval"
            )

        self.ts_id = ts_id
        self.currency = parts[1]
        self.interval = parts[2]

        try:
            self.instrument_type = InstrumentTypeConfig(parts[0])
        except ValueError:
            raise ValueError(f"Unknown instrument type: {parts[0]}")

        self.timestampColumn = timestampColumn
        self.targetColumn = targetColumn

        # Handle database connection
        self.db_session = db_session

        # Normalize forceRecalculate parameter
        self.forceRecalculate = (
            TimeSeriesPreprocessingConfig.normalize_force_recalculate(forceRecalculate)
        )
        self.createNewVersion = createNewVersion

        # Initialize property manager
        self.propertyManager = PropertyManager(
            interval=self.interval, db_session=self.db_session
        )

        # 🔍 DEBUG TRACING: Initialization (temporary helper)
        self._trace_helper = None
        self.enable_tracing = True
        if self.enable_tracing:
            from pipeline.helpers.data_trace import DataTraceHelper
            from pipeline.timeSeriesProcessing.baseModule.baseConfigAdapter import (
                BaseConfigAdapter,
            )
            from pipeline.timeSeriesProcessing.baseModule.baseMethod import (
                BaseTimeSeriesMethod,
            )
            from pipeline.timeSeriesProcessing.baseModule.baseProcessor import (
                BaseProcessor,
            )

            self._trace_helper = DataTraceHelper(base_dir="data/traces")
            self._trace_helper.start_run()

            # Inject into base classes for global access
            BaseProcessor._trace_helper = self._trace_helper  # type: ignore
            BaseTimeSeriesMethod._trace_helper = self._trace_helper  # type: ignore
            BaseConfigAdapter._trace_helper = self._trace_helper  # type: ignore

            # NOTE: For tracing in specific algorithms add _trace_helper = None
            # to classes algorithmAnalyzer, algorithmPeriodicityDetector, etc.

            logging.info(
                f"🔍 {self.__str__()} Tracing enabled: {self._trace_helper.get_run_dir()}"
            )

        # Create processing pipeline
        self.pipeline = self._createPipeline()

        # Track properties
        self.propertySources = {}
        self.currentProperties = {
            "ts_id": self.ts_id,
            "calculated_at": datetime.now(),
            "is_active": 1,
        }

        logging.info(
            f"{self.__str__()} TimeSeriesPreprocessor initialized for {ts_id}"
        )

    def _createPipeline(self):
        """Create processing pipeline"""
        pipeline = ProcessingPipeline()

        # Get all properties once
        prepared_props = self._prepare_all_properties()

        # Add processors
        # 1. Analyzer (basic time series properties)
        pipeline.addProcessor(
            AnalysisProcessor(
                ts_id=self.ts_id,
                currency=self.currency,
                interval=self.interval,
                instrument_type=self.instrument_type,
                targetColumn=self.targetColumn,
                properties=prepared_props["analyzer"],
                analyzerConfig=None,  # if None then auto-select parameters for internal methods
                fallbackBehavior="error",
            )
        )

        # 2. Periodicity detector
        pipeline.addProcessor(
            PeriodicityDetectorProcessor(
                ts_id=self.ts_id,
                currency=self.currency,
                interval=self.interval,
                instrument_type=self.instrument_type,
                targetColumn=self.targetColumn,
                properties=prepared_props["periodicity"],
                detectorConfig=None,  # if None then auto-select parameters for internal methods
                fallbackBehavior="error",
            )
        )

        # 3. Decomposition
        pipeline.addProcessor(
            DecompositionProcessor(
                ts_id=self.ts_id,
                currency=self.currency,
                interval=self.interval,
                instrument_type=self.instrument_type,
                targetColumn=self.targetColumn,
                properties=prepared_props["decomposition"],
                decompositionConfig=None,  # auto-select parameters for internal methods
                fallbackBehavior="error",
            )
        )

        # 4. Outlier detection # ! TODO : status in progress , continue designing this module
        # pipeline.addProcessor(
        #     OutlierDetectionProcessor(
        #         ts_id=self.ts_id,
        #         currency=self.currency,
        #         interval=self.interval,
        #         instrument_type=self.instrument_type,
        #         targetColumn=self.targetColumn,
        #         properties=prepared_props["outlier_detection"],
        #         config=None,  # auto-select parameters for internal methods
        #         fallbackBehavior="error",
        #     )
        # )

        logging.info(
            f"{self.__str__()} Pipeline created: AnalysisProcessor → PeriodicityDetectorProcessor → DecompositionProcessor → OutlierDetectionProcessor"
        )

        return pipeline

    def _prepare_all_properties(self):
        """Get analyzer and periodicity properties for processors"""
        try:
            all_props, sources = self.propertyManager.get_properties(
                ts_id=self.ts_id, force_recalculate=self.forceRecalculate
            )

            # Extract analyzer properties
            try:
                analyzer_props = all_props["analyzer"]
            except KeyError:
                analyzer_props = None
                logging.warning(
                    f"{self.__str__()} Analyzer properties not found for {self.ts_id}"
                )

            # Extract periodicity properties
            try:
                periodicity_props = all_props["periodicity"]
            except KeyError:
                periodicity_props = None
                logging.warning(
                    f"{self.__str__()} Periodicity properties not found for {self.ts_id}"
                )

            # Extract decomposition properties
            try:
                decomposition_props = all_props["decomposition"]
            except KeyError:
                decomposition_props = None
                logging.warning(
                    f"{self.__str__()} Decomposition properties not found for {self.ts_id}"
                )

            # Extract outlier_detection properties
            try:
                outlier_detection_props = all_props["outlier_detection"]
            except KeyError:
                outlier_detection_props = None
                logging.warning(
                    f"{self.__str__()} OutlierDetection properties not found for {self.ts_id}"
                )

            return {
                "analyzer": analyzer_props,
                "periodicity": periodicity_props,
                "decomposition": decomposition_props,
                "outlier_detection": outlier_detection_props,
                "sources": sources,
            }
        except Exception as e:
            logging.error(f"Failed to prepare properties: {e}")
            return {
                "analyzer": None,
                "periodicity": None,
                "decomposition": None,
                "outlier_detection": None,
                "sources": {},
            }

    def process(self, df: pd.DataFrame):
        """Process time series"""
        try:
            # Create context
            context = {
                "propertySources": self.propertySources,  # property source: db, calculated
                "currentProperties": self.currentProperties,  # new or retrieved from DB properties
                "targetColumn": self.targetColumn,

                "ts_id": self.ts_id,
                "currency": self.currency,
                "interval": self.interval,
                "instrument_type": self.instrument_type,
            }

            # 🔍 TRACE: Save pipeline context
            if self._trace_helper:
                self._trace_helper.save_context(
                    {
                        "ts_id": self.ts_id,
                        "currency": self.currency,
                        "interval": self.interval,
                        "instrument_type": self.instrument_type.value,
                        "targetColumn": self.targetColumn,
                        "timestampColumn": self.timestampColumn,
                    },
                    "pipeline_config",
                )

            # Validate input data
            self._validateTimestampAndTarget(df)

            # Set timestamp index
            if self.timestampColumn != df.index.name:
                df.set_index(self.timestampColumn, inplace=True)

            # 🔍 TRACE: Input data
            if self._trace_helper:
                self._trace_helper.save_df(df, "00_input_raw")

            # Run
            processedDf, finalContext = self.pipeline.process(
                df, context, self._trace_helper
            )

            # Check for critical errors from pipeline
            if finalContext.get("error"):
                error_info = finalContext["error"]
                error_msg = f"Critical error in pipeline at stage '{error_info['stage']}': {error_info['message']}"
                logging.error(f"{self.__str__()} {error_msg}")
                raise RuntimeError(error_msg)

            # Update sources and current properties with explicit key handling
            try:
                self.propertySources = finalContext["propertySources"]
            except KeyError:
                self.propertySources = {}
                logging.debug(
                    f"{self.__str__()} PropertySources not found in final context"
                )

            try:
                self.currentProperties.update(finalContext["currentProperties"])
            except KeyError:
                logging.debug(
                    f"{self.__str__()} CurrentProperties not found in final context"
                )

            # Save properties if necessary
            self._savePropertiesIfNeeded()

            # Get results
            scaled_column = f"{self.targetColumn}_scaled"
            if scaled_column in processedDf.columns:
                y = processedDf[scaled_column]
                X = processedDf.drop(columns=[scaled_column])
                return X, y
            else:
                logging.warning(
                    f"{self.__str__()} Column {scaled_column} not found in processing results"
                )
                return processedDf, None

        finally:
            # 🔍 TRACE: Finalization
            if self._trace_helper:
                try:
                    self._trace_helper.finalize()
                    logging.info(
                        f"🔍 {self.__str__()} Trace completed: {self._trace_helper.get_run_dir()}"
                    )
                except Exception as e:
                    logging.warning(f"Failed to finalize trace: {e}")

            # Close database session
            self.db_session.close()

    def _validateTimestampAndTarget(self, df: pd.DataFrame) -> None:
        """Validate input data"""
        if (
            self.timestampColumn not in df.columns
            and df.index.name != self.timestampColumn
        ):
            raise ValueError(
                f"Timestamp column '{self.timestampColumn}' is missing."
            )

        if self.targetColumn not in df.columns:
            raise ValueError(f"Target column '{self.targetColumn}' is missing.")

    def _savePropertiesIfNeeded(self) -> None:
        """Save properties to DB if necessary"""
        # Logic to determine if properties need to be saved
        shouldSave = False
        saveReason = ""

        # 1. If forceRecalculate explicitly specified
        if any(self.forceRecalculate.values()):
            shouldSave = True
            saveReason = "forceRecalculate specified"

        # 2. If some properties were recalculated
        elif any(
            source == PropertySourceConfig.CALCULATED
            for source in self.propertySources.values()
        ):
            calculatedGroups = [
                group
                for group, source in self.propertySources.items()
                if source == PropertySourceConfig.CALCULATED
            ]
            if calculatedGroups:
                shouldSave = True
                saveReason = f"Groups were recalculated: {calculatedGroups}"

        # 3. If new version creation explicitly requested
        elif self.createNewVersion:
            shouldSave = True
            saveReason = "New version creation requested"

        if shouldSave:
            logging.info(f"{self.__str__()} Saving properties, reason: {saveReason}")
            self.propertyManager.save_properties(
                ts_id=self.ts_id,
                properties=self.currentProperties,  # ⭐ THIS DATA GOES TO DB ⭐
                is_create_new_version=self.createNewVersion,
            )
        else:
            logging.info(f"{self.__str__()} No need to update properties in DB")

    def __str__(self):
        return "[Preprocessor][timeSeriesProcessing/preprocessor.py]"