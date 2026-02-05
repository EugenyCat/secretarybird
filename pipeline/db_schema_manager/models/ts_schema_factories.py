from pipeline.helpers.configs import FeatureConfig

# Time series tables use schema factories while ORM tables use SQLAlchemy models
class TimeSeriesSchemaFactory:
    """
        Base interface for schema factories.

        ntk: Time series tables use schema factories while ORM tables use SQLAlchemy models.

    """

    def create_schema(self, ts_id, processing_stage):
        """Creates a schema for the given time series ID and processing stage"""
        raise NotImplementedError("Subclasses must implement this method")


class CryptoRawSchemaFactory(TimeSeriesSchemaFactory):
    """
        Factory for raw crypto time series schemas
    """

    def create_schema(self, ts_id, processing_stage='raw'):
        """Returns a schema for raw crypto data"""

        # Standard columns for raw crypto data
        return [
            ('Open_time', 'DateTime'),
            ('Open', 'Float64'),
            ('High', 'Float64'),
            ('Low', 'Float64'),
            ('Close', 'Float64'),
            ('Volume', 'Float64'),
            ('Close_time', 'DateTime'),
            ('Quote_asset_volume', 'Float64'),
            ('Number_of_trades', 'UInt32'),
            ('Taker_buy_base_asset_volume', 'Float64'),
            ('Taker_buy_quote_asset_volume', 'Float64'),
            ('ts_id', 'String'),
            ('Source', 'String')
        ]


class CryptoTransformedSchemaFactory(TimeSeriesSchemaFactory):
    """
        Factory for transformed crypto time series schemas (processing_stage='transformed').
    """

    def create_schema(self, ts_id):
        """Creates a schema for transformed crypto data"""

        # Extract interval from ts_id
        parts = ts_id.split('_')
        interval = parts[-1]

        # Base columns (essential metadata)
        columns = [
            ('ts_id', 'String'),  # Unified time series identifier
            ('Open_time', 'DateTime'),
            ('Open_original', 'Float64'),  # Original value for reference
            ('Open_removed_outliers', 'String'),  # Value after outlier removal
            ('Open_scaled', 'Float64'),  # Scaled value for models
        ]

        # Add feature columns from centralized configuration
        feature_columns = FeatureConfig.get_feature_schema_columns(interval)
        columns.extend(feature_columns)

        # Add schema version metadata # todo: do I need it ? Remove if deprecated
        #columns.append(('schema_version', 'UInt32'))

        return columns




# Template for future implementations, e.g., for STOCKS data
class StocksRawSchemaFactory(TimeSeriesSchemaFactory):
    """Factory for raw stocks time series schemas"""

    def create_schema(self, ts_id, processing_stage='raw'):
        """Creates a schema for raw stocks data"""
        # Example columns for stocks (similar to crypto but may have different fields)
        columns = [
            ('Open_time', 'DateTime'),
            ('Open', 'Float64'),
            ('High', 'Float64'),
            ('Low', 'Float64'),
            ('Close', 'Float64'),
            ('Volume', 'UInt64'),
            ('Adjusted_close', 'Float64'),  # Specific to stocks
            ('ts_id', 'String'),
            ('Source', 'String DEFAULT \'yahoo_finance\'')  # Different default source
        ]

        return {
            'ts_id': ts_id,
            'processing_stage': processing_stage,
            'columns': columns
        }


class StocksTransformedSchemaFactory(TimeSeriesSchemaFactory):
    """Factory for transformed stocks time series schemas"""

    def create_schema(self, ts_id, processing_stage='transformed'):
        """Creates a schema for transformed stocks data"""
        # Similar to CryptoTransformedSchemaFactory but may have stock-specific features
        # For now, we'll reuse the crypto implementation
        crypto_factory = CryptoTransformedSchemaFactory()
        schema = crypto_factory.create_schema(ts_id, processing_stage)

        # Add or modify stock-specific columns here if needed
        # e.g., add columns for stock-specific indicators

        return schema