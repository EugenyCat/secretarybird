from pipeline.helpers.protocols import TimeSeriesTransformProcessorProtocol
from pipeline.timeSeriesProcessing.timeSeriesAlgorithms.outliers import OutlierDetector
from pipeline.helpers.configs import PropertySourceConfig
from typing import Dict, Any, Tuple
import pandas as pd
import logging
import numpy as np


class OutlierProcessor(TimeSeriesTransformProcessorProtocol):
    """Processor for outlier detection and removal"""

    def __init__(self, targetColumn, interval, propertyManager=None, max_outlier_ratio=0.3):
        """
        Initialize outlier removal processor

        Args:
            targetColumn: Column with data for processing
            interval: Data interval ('1h', '1d', etc.)
            propertyManager: Property manager for saving/loading parameters
            max_outlier_ratio: Maximum outlier ratio (0.0-1.0)
        """
        self.targetColumn = targetColumn
        self.interval = interval
        self.propertyManager = propertyManager
        self.max_outlier_ratio = max_outlier_ratio
        self.outlierDetector = OutlierDetector(interval=interval, max_outlier_ratio=max_outlier_ratio)

    def process(self, data: pd.DataFrame, context: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Detect and remove outliers from time series with improved error handling

        Args:
            data: DataFrame with data
            context: Processing context with metadata

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Processed data and updated context
        """
        try:
            # Get parameters from context
            tsId = context.get('tsId')
            forceRecalculate = context.get('forceRecalculate', {}).get('outlier', False)

            is_stationary = False  # Default value
            if 'currentProperties' in context and 'analyzer' in context['currentProperties']:
                is_stationary = context['currentProperties']['analyzer'].get('stationary', False)

            # Validate input data
            if self.targetColumn not in data.columns:
                logging.error(f"Column {self.targetColumn} is missing in data")
                return data, context

            if data[self.targetColumn].isna().all():
                logging.warning(f"Column {self.targetColumn} contains only NA values")
                # Create column copy without processing
                data[f'{self.targetColumn}_cleaned'] = data[self.targetColumn]
                return data, context

            # Basic data quantity check
            if len(data) < 3:  # Minimum quantity for processing
                logging.warning(f"Insufficient data for outlier processing ({len(data)} points)")
                # Create column copy without processing
                data[f'{self.targetColumn}_cleaned'] = data[self.targetColumn]
                return data, context

            # Attempt to retrieve properties from storage
            existing_props = {}
            method = 'auto'  # Use auto-select method by default
            contamination = None
            n_neighbors = None
            is_combined = None

            # Get saved properties if available
            if self.propertyManager and tsId and not forceRecalculate:
                try:
                    props, sources = self.propertyManager.get_properties(
                        ts_id=tsId,
                        groups=['outlier'],
                        force_recalculate={'outlier': forceRecalculate}
                    )
                    existing_props = props.get('outlier', {})

                    # If properties exist, use their parameters
                    if existing_props:
                        # Get outlier detection method
                        method = existing_props.get('outlier_detection_method', method)

                        # Get contamination values depending on method
                        if method == 'isolation_forest' and 'contamination_isolation_forest' in existing_props:
                            contamination = existing_props['contamination_isolation_forest']
                        elif method == 'lof' and 'contamination_lof' in existing_props:
                            contamination = existing_props['contamination_lof']
                            # For LOF also need n_neighbors
                            if 'n_neighbors_lof' in existing_props:
                                n_neighbors = existing_props['n_neighbors_lof']

                        # Get combined method flag, but with check for short data
                        is_combined = existing_props.get('is_combined')

                        # Update context
                        context['propertySources'] = context.get('propertySources', {})
                        context['propertySources']['outlier'] = sources.get('outlier')
                        context['currentProperties'] = context.get('currentProperties', {})
                        context['currentProperties']['outlier'] = existing_props
                except Exception as e:
                    logging.error(f"Error retrieving properties from propertyManager: {e}")

            # Perform outlier removal with error handling
            logging.info(
                f"Removing outliers: method={method}, is_combined={is_combined}, interval={self.interval}, data_length={len(data)}")

            try:
                # Process outliers accounting for missing values
                processed_data = self._process_series(
                    data[self.targetColumn],
                    method=method,
                    is_combined=is_combined,
                    contamination=contamination,
                    n_neighbors=n_neighbors,
                    is_stationary=is_stationary
                )

                # Add cleaned series to data
                data[f'{self.targetColumn}_cleaned'] = processed_data

                # Get processing metadata
                outlier_metadata = self.outlierDetector.get_metadata()

                # Form results
                outlier_props = {
                    'outlier_detection_method': outlier_metadata.get('outlier_detection_method'),
                    'contamination_isolation_forest': outlier_metadata.get('contamination_isolation_forest'),
                    'contamination_lof': outlier_metadata.get('contamination_lof'),
                    'n_neighbors_lof': outlier_metadata.get('n_neighbors_lof'),
                    'outlier_ratio': outlier_metadata.get('outlier_ratio', 0.0),
                    'is_combined': outlier_metadata.get('is_combined')
                }

                # If new calculation was performed, update properties
                if forceRecalculate or not existing_props:
                    context['propertySources'] = context.get('propertySources', {})
                    context['propertySources']['outlier'] = PropertySourceConfig.CALCULATED
                    context['currentProperties'] = context.get('currentProperties', {})
                    context['currentProperties']['outlier'] = outlier_props

            except Exception as e:
                logging.error(f"Error during outlier removal: {e}")
                # In case of error, return original data
                data[f'{self.targetColumn}_cleaned'] = data[self.targetColumn]

                # Add error information to context
                context['errors'] = context.get('errors', {})
                context['errors']['outlier'] = str(e)

            return data, context

        except Exception as e:
            logging.error(f"Critical error in outlier processor: {e}")
            # In case of critical error, return original data unchanged
            try:
                if self.targetColumn in data.columns:
                    data[f'{self.targetColumn}_cleaned'] = data[self.targetColumn]
            except:
                pass  # Ignore errors when trying to save original data

            return data, context

    def _process_series(self, series: pd.Series, method: str = 'auto',
                        is_combined: int = None, contamination=None, n_neighbors=None, is_stationary=False) -> pd.Series:
        """
        Process time series accounting for missing values

        Args:
            series: Original time series
            method: Outlier detection method
            is_combined: Combined method flag
            contamination: Outlier ratio parameter
            n_neighbors: Number of neighbors for LOF
            is_stationary: series stationarity (expected to be determined in analyzer or from db)

        Returns:
            pd.Series: Processed series
        """
        # Check for missing values
        has_na = series.isna().any()

        if has_na:
            # Save NA value indices
            na_indices = series.isna()
            # For processing use only non-NA values
            valid_series = series.dropna()

            if len(valid_series) < 3:
                logging.warning(f"Insufficient valid data for outlier processing ({len(valid_series)} points)")
                return series  # Return original data

            # Process only valid data
            cleaned_valid = self.outlierDetector.remove_outliers(
                valid_series,
                method=method,
                is_combined=is_combined,
                contamination=contamination,
                n_neighbors=n_neighbors,
                is_stationary=is_stationary
            )

            # Create full series preserving NA in their positions
            cleaned_series = pd.Series(index=series.index, dtype=series.dtype)
            cleaned_series.loc[~na_indices] = cleaned_valid.values
            cleaned_series.loc[na_indices] = np.nan

            return cleaned_series
        else:
            # If no NA, process entire series
            return self.outlierDetector.remove_outliers(
                series,
                method=method,
                is_combined=is_combined,
                contamination=contamination,
                n_neighbors=n_neighbors,
                is_stationary=is_stationary
            )