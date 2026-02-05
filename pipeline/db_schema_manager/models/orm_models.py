from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base, declared_attr

# Workaround for automatic generation of migration version files
__all__ = [
    "TimeSeriesDefinition",
    "TSPreprocessingProperties",
    "Models",
    "ModelTraining",
    "ModelQuality",
    "Migration",
    "SchemaVersion",
    "Schema",
]

# Time series tables use schema factories while ORM tables use SQLAlchemy models
Base = declarative_base()


class ClickHouseModelMixin:
    """
    Mixin for ClickHouse models providing common attributes and settings.
    Use in composition with SQLAlchemy models.
    """

    # ClickHouse attributes with default values
    __clickhouse_engine__ = "MergeTree()"
    __clickhouse_settings__ = {"index_granularity": 8192}
    __clickhouse_partition_by__ = None
    __clickhouse_order_by__ = []


class TimeSeriesDefinition(ClickHouseModelMixin, Base):
    __tablename__ = "time_series_definition"
    __table_args__ = {"schema": "ConfigKernel"}
    __clickhouse_order_by__ = ["ts_id"]

    # Column definitions
    ts_id = Column(String, primary_key=True)
    source = Column(String, nullable=False)
    currency = Column(String, nullable=False)
    interval = Column(String, nullable=False)
    database_name = Column(String, nullable=False)
    is_active = Column(Integer, nullable=False, default=1)
    reference_asset_id = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=True)


import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base


class TSPreprocessingProperties(ClickHouseModelMixin, Base):
    __tablename__ = "ts_preprocessing_properties"
    __table_args__ = {"schema": "ConfigKernel"}
    __clickhouse_order_by__ = ["ts_id", "property_id"]
    __clickhouse_partition_by__ = "toYYYYMM(calculated_at)"

    # Unique identifier of the time series
    ts_id = Column(String, primary_key=True, doc="Time series identifier")
    # Unique identifier for the preprocessing properties
    property_id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        doc="Unique identifier for preprocessing properties",
    )
    # Timestamp when the preprocessing properties were calculated
    calculated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.now,
        doc="Timestamp of property calculation",
    )
    # Indicates if the record is currently active
    is_active = Column(
        Integer, nullable=False, default=1, doc="Flag indicating active record"
    )

    # ----------------- General Properties -----------------
    length = Column(Integer, nullable=True, doc="Length of the time series")
    missing_ratio = Column(
        Float, nullable=True, doc="Ratio of missing values in the series"
    )
    missing_values = Column(
        Integer, nullable=True, doc="Number of missing values in the series"
    )
    volatility = Column(Float, nullable=True, doc="Volatility of the time series")
    skewness = Column(
        Float, nullable=True, doc="Skewness of the time series distribution"
    )
    kurtosis = Column(
        Float, nullable=True, doc="Kurtosis of the time series distribution"
    )
    zscore_outliers = Column(
        Integer, nullable=True, doc="Number of outliers detected by z-score method"
    )
    iqr_outliers = Column(
        Integer, nullable=True, doc="Number of outliers detected by IQR method"
    )
    mad_outliers = Column(
        Integer, nullable=True, doc="Number of outliers detected by MAD method"
    )
    estimated_trend_strength = Column(
        Float, nullable=True, doc="Strength of the trend in the series"
    )
    noise_level = Column(Float, nullable=True, doc="Noise level of the series")
    qq_correlation = Column(Float, nullable=True, doc="Q-Q correlation statistic")

    # Indicates whether the time series is stationary
    is_stationary = Column(
        Integer, nullable=False, default=0, doc="Stationarity flag of the series"
    )
    # P-value of the Augmented Dickey-Fuller (ADF) test
    adf_pvalue = Column(Float, nullable=True, doc="P-value of the ADF test")
    # P-value of the KPSS stationarity test
    kpss_pvalue = Column(Float, nullable=True, doc="P-value of the KPSS test")
    # Autocorrelation value at lag 1
    lag1_autocorrelation = Column(Float, nullable=True, doc="Lag-1 autocorrelation")
    # Coefficient of variation of the rolling mean
    rolling_mean_cv = Column(
        Float, nullable=True, doc="Rolling mean coefficient of variation"
    )
    # Coefficient of variation of the rolling standard deviation
    rolling_std_cv = Column(
        Float, nullable=True, doc="Rolling standard deviation coefficient of variation"
    )
    # Ratio of detected outliers to total number of points
    outlier_ratio = Column(Float, nullable=True, doc="Outlier ratio in the series")
    # Data quality score (from 0 to 1, float)
    data_quality_score = Column(
        Float, nullable=True, doc="Generalized time series quality score (0-1)"
    )
    # Time series type (e.g.: 'stationary_no_trend', 'non_stationary_strong_trend', etc.)
    series_type = Column(
        String, nullable=True, doc="Time series type classification"
    )

    # ----------------- Catch22 Features -----------------
    # Catch22 (CAnonical Time-series CHaracteristics) - 22 canonical time series features
    # Source: https://github.com/DynamicsAndNeuralSystems/catch22

    # Distribution shape features
    c22_mode_5 = Column(Float, nullable=True, doc="5-bin histogram mode")
    c22_mode_10 = Column(Float, nullable=True, doc="10-bin histogram mode")

    # Extreme event timing features
    c22_outlier_timing_pos = Column(Float, nullable=True, doc="Positive outlier timing")
    c22_outlier_timing_neg = Column(Float, nullable=True, doc="Negative outlier timing")

    # Linear autocorrelation features
    c22_acf_timescale = Column(
        Float, nullable=True, doc="First 1/e crossing of the ACF"
    )
    c22_acf_first_min = Column(Float, nullable=True, doc="First minimum of the ACF")
    c22_low_freq_power = Column(
        Float, nullable=True, doc="Power in lowest 20% frequencies"
    )
    c22_centroid_freq = Column(Float, nullable=True, doc="Centroid frequency")
    c22_ami_timescale = Column(
        Float, nullable=True, doc="First minimum of the AMI function"
    )
    c22_periodicity = Column(Float, nullable=True, doc="Wang's periodicity metric")

    # Nonlinear autocorrelation features
    c22_ami2 = Column(
        Float,
        nullable=True,
        doc="Histogram-based automutual information (lag 2, 5 bins)",
    )
    c22_trev = Column(Float, nullable=True, doc="Time reversibility")

    # Symbolic features
    c22_stretch_high = Column(
        Float, nullable=True, doc="Longest stretch of above-mean values"
    )
    c22_stretch_decreasing = Column(
        Float, nullable=True, doc="Longest stretch of decreasing values"
    )
    c22_entropy_pairs = Column(
        Float, nullable=True, doc="Entropy of successive pairs in symbolized series"
    )
    c22_transition_variance = Column(
        Float, nullable=True, doc="Transition matrix column variance"
    )

    # Incremental differences features
    c22_whiten_timescale = Column(
        Float,
        nullable=True,
        doc="Change in autocorrelation timescale after incremental differencing",
    )
    c22_high_fluctuation = Column(
        Float, nullable=True, doc="Proportion of high incremental changes in the series"
    )

    # Simple forecasting features
    c22_forecast_error = Column(
        Float, nullable=True, doc="Error of 3-point rolling mean forecast"
    )

    # Self-affine scaling features
    c22_rs_range = Column(
        Float,
        nullable=True,
        doc="Rescaled range fluctuation analysis (low-scale scaling)",
    )
    c22_dfa = Column(
        Float, nullable=True, doc="Detrended fluctuation analysis (low-scale scaling)"
    )

    # Other features
    c22_embedding_dist = Column(
        Float,
        nullable=True,
        doc="Goodness of exponential fit to embedding distance distribution",
    )

    # ----------------- Periodicity -----------------
    # Dominant detected period in the series
    main_period = Column(Integer, nullable=True, doc="Detected main period")
    # JSON array of detected periods
    periods = Column(String, nullable=True, doc="JSON array of detected periods")
    # JSON array of confidence scores for detected periods
    period_confidence_scores = Column(
        String, nullable=True, doc="JSON array of confidence scores"
    )
    # Method used for periodicity detection (e.g., 'ensemble', 'acf', 'spectral')
    periodicity_detection_method = Column(
        String, nullable=True, doc="Method used for periodicity detection"
    )
    # Seasonal strength measured by STL verification
    stl_seasonal_strength = Column(
        Float, nullable=True, doc="Seasonal strength from STL verification"
    )
    # Trend strength measured by STL verification
    stl_trend_strength = Column(
        Float, nullable=True, doc="Trend strength from STL verification"
    )
    # JSON containing detailed results from all detection methods
    periodicity_method_results = Column(
        String, nullable=True, doc="JSON with detailed results from detection methods"
    )
    # JSON containing autocorrelation function (ACF) values for various lags
    acf_values = Column(
        String, nullable=True, doc="JSON with ACF values for various lags"
    )
    # Additional periodicity fields (fallback logic)
    suggested_periods = Column(
        String, nullable=True, doc="JSON: Suggested periods based on heuristics"
    )
    detection_status = Column(
        String, nullable=True, doc="Detection status: detected/not_detected/unknown"
    )
    periodicity_quality_score = Column(
        Float, nullable=True, doc=""
    )

    # ----------------- Decomposition -----------------
    # Best detected decomposition method ('STL', 'SeasonalDecompose', etc.)
    decomposition_method = Column(
        String, nullable=True, doc="Optimal decomposition method"
    )
    # Baseline quality score from diagnostic STL decomposition
    baseline_quality = Column(
        Float, nullable=True, doc="Quality score from baseline STL decomposition"
    )
    # Type of decomposition model used
    model_type = Column(
        String, nullable=True, doc="Type of decomposition model (additive/multiplicative)"
    )
    # Strength of trend component from decomposition
    trend_strength = Column(
        Float, nullable=True, doc="Strength of trend component (0-1)"
    )
    # Strength of seasonal component from decomposition
    seasonal_strength = Column(
        Float, nullable=True, doc="Strength of seasonal component (0-1)"
    )
    # Strength of residual component from decomposition
    residual_strength = Column(
        Float, nullable=True, doc="Strength of residual component (0-1)"
    )
    # Overall quality score of decomposition
    quality_score = Column(
        Float, nullable=True, doc="Overall decomposition quality score"
    )
    # Mean squared error of decomposition reconstruction
    reconstruction_error = Column(
        Float, nullable=True, doc="MSE between original and reconstructed series"
    )
    # MSTL stability metrics convergence status
    mstl_stability_metrics_converged = Column(
        Integer, nullable=True, doc="MSTL convergence flag (1=converged, 0=not converged)"
    )
    # JSON array of MSTL parameter corrections applied
    mstl_corrections_applied = Column(
        String, nullable=True, doc="JSON array of parameter corrections applied in MSTL"
    )
    # === Fourier Decomposition Metrics ===
    fourier_n_harmonics = Column(Integer, nullable=True, doc="Number of significant harmonics in Fourier decomposition")
    fourier_spectral_entropy = Column(Float, nullable=True,
                                      doc="Spectral entropy measure of signal chaos in Fourier analysis")
    fourier_harmonic_strength = Column(Float, nullable=True,
                                       doc="Overall strength of harmonic components in Fourier decomposition")
    # === SSA Decomposition Metrics ===
    ssa_window_length = Column(Integer, nullable=True, doc="Window length of trajectory matrix in SSA decomposition")
    ssa_n_components_used = Column(Integer, nullable=True,
                                   doc="Number of significant SVD components used in SSA reconstruction")
    ssa_variance_explained = Column(Float, nullable=True,
                                    doc="Proportion of variance explained by selected SSA components (0-1)")
    ssa_component_grouping = Column(String, nullable=True,
                                    doc="JSON grouping of SSA components into trend and seasonal parts")
    # === TBATS Decomposition Metrics ===
    tbats_seasonal_periods = Column(String, nullable=True, doc="JSON array of seasonal periods used in TBATS model")
    tbats_box_cox_lambda = Column(Float, nullable=True,
                                  doc="Box-Cox transformation parameter for variance stabilization in TBATS")
    tbats_aic = Column(Float, nullable=True, doc="Akaike Information Criterion for TBATS model quality assessment")
    tbats_arma_order = Column(String, nullable=True, doc="JSON ARMA order (p,q) for error modeling in TBATS")
    tbats_damped_trend = Column(Boolean, nullable=True, doc="Flag indicating damped trend usage in TBATS model")
    # === Prophet Decomposition Metrics ===
    prophet_changepoints_detected = Column(
        Boolean, nullable=True, doc="Flag indicating if structural changepoints were detected in Prophet trend analysis"
    )
    prophet_trend_flexibility = Column(
        Float, nullable=True,
        doc="Quantitative measure of trend variability and changepoint flexibility in Prophet model"
    )
    prophet_seasonality_mode = Column(
        String, nullable=True, doc="Selected seasonality mode (additive/multiplicative) in Prophet decomposition"
    )
    prophet_cross_validation_mape = Column(
        Float, nullable=True, doc="Cross-validation MAPE for Prophet forecast accuracy assessment"
    )
    prophet_cross_validation_rmse = Column(
        Float, nullable=True, doc="Cross-validation RMSE for Prophet forecast accuracy assessment"
    )
    prophet_bayesian_uncertainty = Column(
        Float, nullable=True, doc="Bayesian uncertainty measure from Prophet probabilistic forecasting"
    )
    prophet_trend_changepoint_dates = Column(
        String, nullable=True, doc="JSON array of detected structural changepoint dates in Prophet trend analysis"
    )
    prophet_component_importance = Column(
        String, nullable=True,
        doc="JSON object with relative importance scores of trend, seasonality, and holiday components"
    )
    # === N-BEATS Neural Decomposition Metrics ===
    nbeats_model_type = Column(
        String, nullable=True, doc="Type of N-BEATS model used (pytorch_nbeats for neural network or linear_fallback)"
    )
    nbeats_training_loss = Column(
        Float, nullable=True, doc="Final training MSE loss achieved during N-BEATS neural network optimization"
    )
    nbeats_convergence_achieved = Column(
        Boolean, nullable=True, doc="Flag indicating successful convergence of N-BEATS neural network training"
    )
    nbeats_harmonic_complexity = Column(
        Float, nullable=True,
        doc="Quantitative measure of harmonic pattern complexity detected by N-BEATS seasonal stacks"
    )
    nbeats_seasonal_harmonics_used = Column(
        Integer, nullable=True, doc="Number of Fourier harmonics adaptively selected by N-BEATS seasonal blocks"
    )
    nbeats_architecture_efficiency = Column(
        Float, nullable=True,
        doc="Architecture efficiency metric measuring N-BEATS model performance vs computational cost"
    )


    # ----------------- Outlier Detection -----------------
    # Best contamination value (proportion of outliers) for IsolationForest
    contamination_isolation_forest = Column(
        Float, nullable=True, doc="Optimal contamination for IsolationForest"
    )
    # Best contamination value for Local Outlier Factor (LOF)
    contamination_lof = Column(
        Float, nullable=True, doc="Optimal contamination for LOF"
    )
    # Optimal number of neighbors for LOF method
    n_neighbors_lof = Column(
        Integer, nullable=True, doc="Optimal number of neighbors for LOF"
    )
    is_combined = Column(
        Integer, nullable=False, default=0, doc="Combined method for outlier"
    )
    # Preferred method for outlier detection ('IsolationForest', 'LOF', etc.)
    outlier_detection_method = Column(
        String, nullable=True, doc="Preferred outlier detection method"
    )

    # ----------------- Scaling -----------------
    # Type of scaler applied ('minmax', 'standard', 'robust', 'quantile', 'log')
    scaler_type = Column(String, nullable=True, doc="Scaler type")
    # JSON with parameters used for the scaler
    scaler_params = Column(String, nullable=True, doc="JSON with scaler parameters")
    # Serialized (Base64 encoded) scaler model
    scaler_serialized = Column(
        String, nullable=True, doc="Serialized scaler model (Base64 encoded)"
    )
    # JSON with statistics of the original time series
    original_stats = Column(
        String, nullable=True, doc="JSON with original series statistics"
    )
    # JSON with statistics of the scaled time series
    scaled_stats = Column(
        String, nullable=True, doc="JSON with scaled series statistics"
    )

    # ----------------- Max Age Configuration -----------------
    # Maximum age for each property group (in days)
    max_age_analyzer = Column(
        Integer, nullable=True, doc="Maximum age in days for analyzer properties"
    )
    max_age_periodicity = Column(
        Integer, nullable=True, doc="Maximum age in days for periodicity properties"
    )
    max_age_outlier_detection = Column(
        Integer,
        nullable=True,
        doc="Maximum age in days for outlier detection properties",
    )
    max_age_scaling = Column(
        Integer, nullable=True, doc="Maximum age in days for scaling properties"
    )
    max_age_decomposition = Column(
        Integer, nullable=True, doc="Maximum age in days for decomposition properties"
    )

    # ----------------- Override Controls -----------------
    # Flag to force recalculation for each property group
    force_recalc_analyzer = Column(
        Integer,
        nullable=False,
        default=0,
        doc="Flag to force recalculation of analyzer properties",
    )
    force_recalc_periodicity = Column(
        Integer,
        nullable=False,
        default=0,
        doc="Flag to force recalculation of periodicity properties",
    )
    force_recalc_outlier = Column(
        Integer,
        nullable=False,
        default=0,
        doc="Flag to force recalculation of outlier properties",
    )
    force_recalc_scaling = Column(
        Integer,
        nullable=False,
        default=0,
        doc="Flag to force recalculation of scaling properties",
    )
    force_recalc_decomposition = Column(
        Integer,
        nullable=False,
        default=0,
        doc="Flag to force recalculation of decomposition properties",
    )

    # ----------------- Configuration Storage -----------------
    # JSON field storing complete analyzer method configurations used for processing
    config_analyzer = Column(
        String,
        nullable=True,
        doc="JSON field storing complete analyzer method configurations",
    )

    config_periodicity = Column(
        String,
        nullable=True,
        doc="JSON field storing complete periodicity method configurations",
    )

    config_decomposition = Column(
        String,
        nullable=True,
        doc="JSON field storing complete decomposition method configurations",
    )

    config_outlier = Column(
        String,
        nullable=True,
        doc="JSON field storing complete outlier method configurations",
    )

    config_feature_engineering = Column(
        String,
        nullable=True,
        doc="JSON field storing complete feature engineering method configurations",
    )

    def __repr__(self):
        return (
            f"<TSPreprocessingProperties(ts_id='{self.ts_id}', property_id='{self.property_id}', "
            f"calculated_at='{self.calculated_at}')>"
        )


class Models(ClickHouseModelMixin, Base):
    __tablename__ = "models"
    __table_args__ = {"schema": "ConfigKernel"}

    # Column definitions
    model_id = Column(Integer, primary_key=True)
    model_name = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    model_params = Column(String, nullable=True)  # JSON string with parameters
    description = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=True)


class ModelTraining(ClickHouseModelMixin, Base):
    __tablename__ = "model_training"
    __table_args__ = {"schema": "ConfigKernel"}
    __clickhouse_partition_by__ = "toYYYYMM(training_date)"
    __clickhouse_order_by__ = []

    # Column definitions
    training_id = Column(Integer, primary_key=True)
    model_id = Column(Integer, nullable=False)
    training_currency = Column(String, nullable=False)
    training_interval = Column(String, nullable=False)
    training_interval_New = Column(String, nullable=False)
    training_params = Column(String, nullable=True)  # JSON string with parameters
    score = Column(Float, nullable=True)
    training_date = Column(DateTime, nullable=False)
    model_path = Column(String, nullable=True)


class ModelQuality(ClickHouseModelMixin, Base):
    __tablename__ = "model_quality"
    __table_args__ = {"schema": "ConfigKernel"}
    __clickhouse_partition_by__ = "toYYYYMM(evaluation_date)"

    # Column definitions
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, nullable=False)
    training_id = Column(Integer, nullable=False)
    currency = Column(String, nullable=False)
    interval = Column(String, nullable=False)
    mae = Column(Float, nullable=True)
    mse = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    mape = Column(Float, nullable=True)
    r2 = Column(Float, nullable=True)
    evaluation_date = Column(DateTime, nullable=False)
    additional_metrics = Column(
        String, nullable=True
    )  # JSON string with additional metrics


# Migration tables
class Migration(ClickHouseModelMixin, Base):
    __tablename__ = "migrations"
    __table_args__ = {"schema": "InfraKernel"}

    migration_id = Column(
        String, primary_key=True, doc="Unique migration identifier"
    )
    applied_at = Column(
        DateTime, nullable=False, default=datetime.now, doc="Migration application time"
    )
    description = Column(String, nullable=False, doc="Migration description")
    schema_changes = Column(
        String, nullable=True, doc="JSON string with schema change information"
    )

    def __repr__(self):
        return f"<Migration(migration_id='{self.migration_id}', applied_at='{self.applied_at}', description='{self.description[:30]}...')>"


class SchemaVersion(ClickHouseModelMixin, Base):
    __tablename__ = "schema_versions"
    __table_args__ = {"schema": "InfraKernel"}
    __clickhouse_order_by__ = ["schema_id", "version"]

    # Column definitions
    schema_id = Column(
        String, primary_key=True, doc="Schema identifier (e.g., 'CRYPTO_raw')"
    )
    version = Column(Integer, primary_key=True, doc="Schema version number")
    schema_definition = Column(
        String, nullable=False, doc="JSON string with schema definition"
    )
    changelog = Column(String, nullable=False, doc="Description of changes in this version")
    created_at = Column(
        DateTime, nullable=False, default=datetime.now, doc="Version creation time"
    )

    def __repr__(self):
        return f"<SchemaVersion(schema_id='{self.schema_id}', version={self.version}, created_at='{self.created_at}')>"


class Schema(ClickHouseModelMixin, Base):
    __tablename__ = "schemas"
    __table_args__ = {"schema": "InfraKernel"}

    # Column definitions
    schema_id = Column(String, primary_key=True, doc="Unique schema identifier")
    current_version = Column(
        Integer, nullable=False, default=1, doc="Current active schema version"
    )
    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.now,
        doc="Schema record creation time",
    )
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.now,
        onupdate=datetime.now,
        doc="Last update time",
    )

    def __repr__(self):
        return f"<Schema(schema_id='{self.schema_id}', current_version={self.current_version})>"