from pipeline.helpers.protocols import TimeSeriesTransformProcessorProtocol
from pipeline.timeSeriesProcessing.timeSeriesAlgorithms.features import FeatureGenerator
from pipeline.helpers.configs import PropertySourceConfig
from pipeline.helpers.utils import serialize_object, deserialize_object
from typing import Dict, Any, Tuple
import pandas as pd
import logging
import json

class FeatureEngineeringProcessor(TimeSeriesTransformProcessorProtocol):
    """Processor for feature engineering and scaling"""

    def __init__(self, targetColumn, interval, scalerType='robust', lagFeatures=5, propertyManager=None):
        self.targetColumn = targetColumn
        self.interval = interval
        self.scalerType = scalerType
        self.lagFeatures = lagFeatures
        self.propertyManager = propertyManager
        self.featureGenerator = FeatureGenerator(
            interval=interval,
            scaler_type=scalerType,
            lag_features=lagFeatures,
            target_column=targetColumn
        )

    def process(self, data: pd.DataFrame, context: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Data scaling and feature engineering"""
        tsId = context.get('tsId')
        forceRecalculate = context.get('forceRecalculate', {}).get('scaling', False)

        # Step 1: Attempt to retrieve scaling properties from storage
        existing_scaling_props = {}
        if self.propertyManager and tsId:
            props, sources = self.propertyManager.get_properties(
                ts_id=tsId,
                groups=['scaling'],
                force_recalculate={'scaling': forceRecalculate}
            )
            existing_scaling_props = props.get('scaling', {})

            # If properties exist and scaler is saved, can use it
            if (existing_scaling_props and 'scaler_serialized' in existing_scaling_props and
                    existing_scaling_props['scaler_serialized'] and not forceRecalculate):

                try:
                    # Deserialize scaler and set it in featureGenerator
                    self.featureGenerator.scaler_type = existing_scaling_props['scaler_type']
                    self.featureGenerator.scaler = deserialize_object(existing_scaling_props['scaler_serialized'])

                    logging.info(f"Using existing scaler of type: {self.featureGenerator.scaler_type}")

                    # Update context
                    context['propertySources'] = context.get('propertySources', {})
                    context['propertySources']['scaling'] = sources.get('scaling')
                    context['currentProperties'] = context.get('currentProperties', {})
                    context['currentProperties']['scaling'] = existing_scaling_props
                except Exception as e:
                    logging.warning(f"Scaler deserialization error: {e}. A new one will be created.")
                    forceRecalculate = True

        # Step 2: Determine source column for scaling
        source_column = f'{self.targetColumn}_cleaned'
        if source_column not in data.columns:
            source_column = self.targetColumn
            logging.info(f"Column {self.targetColumn}_cleaned not found, using {self.targetColumn}")

        # Step 3: Scale data (if needed)
        if forceRecalculate or not existing_scaling_props:
            logging.info("Performing data scaling")

            # Record original statistics
            original_stats = {
                'min': float(data[source_column].min()),
                'max': float(data[source_column].max()),
                'mean': float(data[source_column].mean()),
                'std': float(data[source_column].std()),
                'median': float(data[source_column].median())
            }

            # Scale data
            scaled_series = self.featureGenerator.scale_data(data[source_column], log_scaling_stats=True)
            data[f'{self.targetColumn}_scaled'] = scaled_series

            # Scaled data statistics
            scaled_stats = {
                'min': float(scaled_series.min()),
                'max': float(scaled_series.max()),
                'mean': float(scaled_series.mean()),
                'std': float(scaled_series.std())
            }

            # Serialize scaler
            scaler_serialized = None
            if hasattr(self.featureGenerator, 'scaler') and self.featureGenerator.scaler:
                scaler_serialized = serialize_object(self.featureGenerator.scaler)

            # Form results
            scaling_props = {
                'scaler_type': self.featureGenerator.scaler_type,
                'scaler_params': json.dumps(getattr(self.featureGenerator, 'scaler_params', {})),
                'scaler_serialized': scaler_serialized,
                'original_stats': json.dumps(original_stats),
                'scaled_stats': json.dumps(scaled_stats)
            }

            # Update context
            context['propertySources'] = context.get('propertySources', {})
            context['propertySources']['scaling'] = PropertySourceConfig.CALCULATED
            context['currentProperties'] = context.get('currentProperties', {})
            context['currentProperties']['scaling'] = scaling_props
        else:
            # If using existing scaler, still need to apply it to data
            if f'{self.targetColumn}_scaled' not in data.columns:
                scaled_series = self.featureGenerator.scale_data(data[source_column], log_scaling_stats=False)
                data[f'{self.targetColumn}_scaled'] = scaled_series

        # Step 4: Feature generation
        logging.info("Generating features")
        result_df = self.featureGenerator.add_features(data)

        return result_df, context