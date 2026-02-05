-- Create the database to store configuration and metadata for time series and models
CREATE DATABASE IF NOT EXISTS ConfigKernel;


-- TABLE: time_series_definition
-- Defines available time series with source, currency, and interval details
CREATE TABLE IF NOT EXISTS ConfigKernel.time_series_definition (
    ts_id String DEFAULT concat(lower(currency), '_', interval),  -- Unique time series ID
    source String,                   -- Data source (e.g., binance_api)
    currency String,                 -- Currency pair (e.g., btcusdt)
    interval String,                 -- Timeframe (e.g., 1h, 1d)
    database_name String,            -- Target database for this time series
    is_active Boolean DEFAULT true,  -- Whether this time series is active
    created_at DateTime DEFAULT now(),  -- Creation timestamp
    updated_at DateTime DEFAULT now()   -- Last update timestamp
) ENGINE = MergeTree()
ORDER BY (ts_id);


-- Insert initial time series definitions for two currencies with multiple intervals
INSERT INTO ConfigKernel.time_series_definition (source, currency, interval, database_name)
VALUES
('binance_api', 'btcusdt', '1h', 'CRYPTO'),
('binance_api', 'btcusdt', '12h', 'CRYPTO'),
('binance_api', 'btcusdt', '1d', 'CRYPTO'),
('binance_api', 'btcusdt', '3d', 'CRYPTO'),
('binance_api', 'btcusdt', '1w', 'CRYPTO'),
('binance_api', 'btcusdt', '1M', 'CRYPTO'),
('binance_api', 'dotusdt', '1h', 'CRYPTO'),
('binance_api', 'dotusdt', '12h', 'CRYPTO'),
('binance_api', 'dotusdt', '1d', 'CRYPTO'),
('binance_api', 'dotusdt', '3d', 'CRYPTO'),
('binance_api', 'dotusdt', '1w', 'CRYPTO'),
('binance_api', 'dotusdt', '1M', 'CRYPTO');


-- TABLE: models
-- Stores metadata about available models and their hyperparameter configurations
CREATE TABLE ConfigKernel.models (
    model_id UInt64,               -- Unique identifier for the model
    model_name String,             -- Name of the model (e.g., XGBoost, RandomForest)
    hyperparam_space String,       -- Model hyperparameter space (possibly JSON)
    created_at DateTime,           -- When the model was created
    updated_at DateTime            -- Last updated timestamp
) ENGINE = MergeTree()
ORDER BY (model_id);

-- Example DROP statement (possibly used for testing)
DROP TABLE MODEL_REGISTRY.models;


-- TABLE: model_training
-- Stores information about specific model training runs
CREATE TABLE ConfigKernel.model_training (
    training_id UInt64,             -- Unique ID for the training session
    model_id UInt64,                -- Foreign key to the models table
    model_params String,            -- Final model parameters used (can be JSON)
    score Float32,                  -- Evaluation metric (e.g., accuracy, F1)
    training_currency String,       -- Currency used during training
    training_interval String,       -- Timeframe used during training
    created_at DateTime,            -- When the training started
    training_duration UInt32        -- Duration in seconds
) ENGINE = MergeTree()
PARTITION BY model_id               -- Partition by model ID
ORDER BY (training_id);


-- TABLE: model_quality
-- Records quality and error information for model predictions
CREATE TABLE ConfigKernel.model_quality (
    quality_id UInt64,             -- Unique ID for the quality record
    model_id UInt64,               -- Foreign key to the model
    forecast_value Float32,        -- Predicted value
    actual_value Float32,          -- Ground truth value
    error Float32,                 -- Forecast error (e.g., absolute/relative diff)
    created_at DateTime            -- Timestamp of prediction evaluation
) ENGINE = MergeTree()
ORDER BY (quality_id);


-- TABLE: migrations
-- Tracks applied schema changes for audit and reproducibility
CREATE TABLE IF NOT EXISTS ConfigKernel.migrations (
    migration_id String,                  -- Unique ID of the migration
    applied_at DateTime DEFAULT now(),   -- When the migration was applied
    description String,                  -- Human-readable description
    schema_changes String,               -- JSON with exact schema diffs
    PRIMARY KEY (migration_id)
) ENGINE = MergeTree();


-- TABLE: schemas
-- Tracks overall schema metadata (e.g., tracking data versioning system)
CREATE TABLE IF NOT EXISTS ConfigKernel.schemas (
    schema_id String,                     -- Schema ID
    current_version UInt32 DEFAULT 1,    -- Current version number
    created_at DateTime DEFAULT now(),   -- When schema was added
    updated_at DateTime DEFAULT now(),   -- Last update timestamp
    PRIMARY KEY (schema_id)
) ENGINE = MergeTree();


-- TABLE: schema_versions
-- Stores all historical versions of a schema for diff and rollback purposes
CREATE TABLE IF NOT EXISTS ConfigKernel.schema_versions (
    schema_id String,                     -- Associated schema ID
    version UInt32,                       -- Specific version number
    schema_definition String,             -- Full schema definition (JSON)
    changelog String,                     -- Description of changes
    created_at DateTime DEFAULT now(),   -- When version was created
    PRIMARY KEY (schema_id, version)
) ENGINE = MergeTree()
ORDER BY (schema_id, version);


-- TABLE: currency_timeframe_raw
-- Stores raw OHLCV + meta time series data
CREATE TABLE IF NOT EXISTS currency_timeframe_raw (
    Open_time DateTime,                      -- Candle open timestamp
    Open Float64,                            -- Opening price
    High Float64,                            -- Highest price
    Low Float64,                             -- Lowest price
    Close Float64,                           -- Closing price
    Volume Float64,                          -- Trade volume
    Close_time DateTime,                     -- Candle close timestamp
    Quote_asset_volume Float64,              -- Volume in quote asset
    Number_of_trades UInt32,                 -- Trade count
    Taker_buy_base_asset_volume Float64,     -- Buy volume in base asset
    Taker_buy_quote_asset_volume Float64,    -- Buy volume in quote asset
    ts_id String,                            -- Unified time series ID
    Source String                            -- Source (e.g., binance)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(Open_time)
ORDER BY (Open_time);


-- TABLE: currency_timeframe_transformed
-- Stores cleaned and processed version of time series data
CREATE TABLE IF NOT EXISTS currency_timeframe_transformed (
    ts_id String,                            -- Unified time series identifier
    Open_time DateTime,                      -- Original timestamp
    Open_original Float64,                   -- Original open value
    Open_removed_outliers String,            -- Processed value (e.g., outlier removed) as string or JSON
    Open_scaled Float64                      -- Scaled value for model input
    -- other lags and windows features depend on timeframe
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(Open_time)
ORDER BY (Open_time);
