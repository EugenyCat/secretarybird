CREATE DATABASE IF NOT EXISTS ConfigKernel;

CREATE DATABASE IF NOT EXISTS InfraKernel;

DROP DATABASE ConfigKernel;

DROP DATABASE InfraKernel;



CREATE TABLE ConfigKernel.models (
    model_id UInt64,             -- Unique identifier of the model
    model_name String,           -- Model name (e.g. 'RandomForest', 'XGBoost')
    hyperparam_space String,     -- Model hyperparameter space (can be stored as JSON string)
    created_at DateTime,         -- Record creation timestamp (model training start time)
    updated_at DateTime          -- Last model update timestamp
) ENGINE = MergeTree()
ORDER BY (model_id);

DROP TABLE MODEL_REGISTRY.models;


CREATE TABLE ConfigKernel.model_training (
    training_id UInt64,             -- Unique training run identifier
    model_id UInt64,                -- Reference to model from models table
    model_params String,            -- Model parameters (can be stored as JSON string)
    score Float32,                  -- Model quality metric (e.g. accuracy, F1 score, etc.)
    training_currency String,       -- Currency used during training
    training_interval String,       -- Time interval used for training
    created_at DateTime,            -- Training start timestamp
    training_duration UInt32        -- Training duration in seconds
) ENGINE = MergeTree()
PARTITION BY model_id              -- Partitioning by model_id
ORDER BY (training_id);


DROP TABLE MODEL_REGISTRY.model_training;




CREATE TABLE ConfigKernel.model_quality (
    quality_id UInt64,              -- Unique model quality record identifier
    model_id UInt64,                -- Reference to model
    forecast_value Float32,         -- Predicted value
    actual_value Float32,           -- Actual observed value
    error Float32,                  -- Prediction error (e.g. forecast minus actual)
    created_at DateTime             -- Quality record timestamp
) ENGINE = MergeTree()
ORDER BY (quality_id);






SELECT *
FROM system.tables
WHERE database = 'CRYPTO';





-- time_series_definition
-- Table for defining available time series
CREATE TABLE IF NOT EXISTS ConfigKernel.time_series_definition (
    ts_id String DEFAULT concat(lower(currency), '_', interval),
    source String,                  -- Data source identifier (e.g. API name)
    currency String,                -- Trading pair or asset symbol
    interval String,                -- Time interval (e.g. 1h, 1d, 1w)
    database_name String,           -- Target database where time series data is stored
    is_active Boolean DEFAULT true, -- Indicates whether the time series is active
    created_at DateTime DEFAULT now(),
    updated_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (ts_id);

-- Indexes for query optimization
ALTER TABLE ConfigKernel.time_series_definition ADD INDEX idx_source (source) TYPE minmax GRANULARITY 8192;
ALTER TABLE ConfigKernel.time_series_definition ADD INDEX idx_currency (currency) TYPE minmax GRANULARITY 8192;
ALTER TABLE ConfigKernel.time_series_definition ADD INDEX idx_interval (interval) TYPE minmax GRANULARITY 8192;

-- Initial seed data
INSERT INTO ConfigKernel.time_series_definition (source, currency, interval, database_name)
VALUES ('binance_api', 'btcusdt', '1h', 'CRYPTO');
INSERT INTO ConfigKernel.time_series_definition (source, currency, interval, database_name)
VALUES ('binance_api', 'btcusdt', '12h', 'CRYPTO');
INSERT INTO ConfigKernel.time_series_definition (source, currency, interval, database_name)
VALUES ('binance_api', 'btcusdt', '1d', 'CRYPTO');
INSERT INTO ConfigKernel.time_series_definition (source, currency, interval, database_name)
VALUES ('binance_api', 'btcusdt', '3d', 'CRYPTO');
INSERT INTO ConfigKernel.time_series_definition (source, currency, interval, database_name)
VALUES ('binance_api', 'btcusdt', '1w', 'CRYPTO');
INSERT INTO ConfigKernel.time_series_definition (source, currency, interval, database_name)
VALUES ('binance_api', 'btcusdt', '1M', 'CRYPTO');
INSERT INTO ConfigKernel.time_series_definition (source, currency, interval, database_name)
VALUES ('binance_api', 'dotusdt', '1h', 'CRYPTO');
INSERT INTO ConfigKernel.time_series_definition (source, currency, interval, database_name)
VALUES ('binance_api', 'dotusdt', '12h', 'CRYPTO');
INSERT INTO ConfigKernel.time_series_definition (source, currency, interval, database_name)
VALUES ('binance_api', 'dotusdt', '1d', 'CRYPTO');
INSERT INTO ConfigKernel.time_series_definition (source, currency, interval, database_name)
VALUES ('binance_api', 'dotusdt', '3d', 'CRYPTO');
INSERT INTO ConfigKernel.time_series_definition (source, currency, interval, database_name)
VALUES ('binance_api', 'dotusdt', '1w', 'CRYPTO');
INSERT INTO ConfigKernel.time_series_definition (source, currency, interval, database_name)
VALUES ('binance_api', 'dotusdt', '1M', 'CRYPTO');


DROP TABLE IF EXISTS ConfigKernel.time_series_definition;
