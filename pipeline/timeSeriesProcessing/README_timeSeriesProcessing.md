# Time Series Processing Module

> Comprehensive time series preprocessing system with three-stage pipeline and adaptive configuration for financial instruments

## 🎯 Purpose

The timeSeriesProcessing module represents an enterprise-grade system for complete financial time series preprocessing. It implements a three-stage pipeline: **Analyzer → Periodicity → Decomposition** with the possibility of expansion to **5-stage** (planned addition of OutlierRemover and FeatureEngineer).

Main task: transform raw time series into structured sets of components and characteristics ready for machine learning or analytics. The system automatically adapts to instrument type (CRYPTO), data frequency, and series characteristics.

## 🔬 Mathematical Foundation

### 3-Stage Pipeline
1. **Analysis Stage**: statistical characteristics, stationarity, outliers, data quality
2. **Periodicity Stage**: ensemble periodicity detection (ACF + Spectral + Wavelet)  
3. **Decomposition Stage**: intelligent decomposition into trend/seasonal/residual components

### Adaptive Configuration System
- **ConfigurationAdapterProtocol**: unified protocol for all modules
- **Frequency classification**: HIGH/MEDIUM/LOW automatically configures parameters
- **Data length adaptation**: TINY→MASSIVE categories affect windows and thresholds
- **Instrument-specific tuning**: CRYPTO receives special corrections

### Property Management System
- **Intelligent caching**: automatic property relevance determination by max_age
- **Group-based storage**: analyzer/periodicity/decomposition properties stored separately
- **Version control**: versioning support and force_recalculate

### Decomposition Methods (7 algorithms)
- **Fourier**: O(n log n) for stationary with clear periodicity
- **SSA**: Singular Spectrum Analysis for non-stationary series
- **TBATS**: multiple seasonality with Box-Cox + ARMA
- **MSTL**: Multiple Seasonal Trend decomposition using Loess
- **RobustSTL**: robust STL decomposition for noisy data
- **Prophet**: Facebook Prophet for trends with holidays
- **N-BEATS**: neural decomposition with harmonic blocks

## 🏗️ Technical Implementation

**Architectural pattern**: Facade + Pipeline + Strategy + Configuration Adapter

**Main classes**:
- `TimeSeriesPreprocessor`: system facade, pipeline coordination
- `ProcessingPipeline`: executor for sequential processor execution
- `PropertyManager`: property caching and persistence management
- `TimeSeriesPreprocessingConfig`: centralized property group configurations

**Key methods**:
- `TimeSeriesPreprocessor.process()`: main entry point for processing
- `ProcessingPipeline.process()`: sequential stage execution
- `PropertyManager.get_properties()`: intelligent property loading from DB
- `build_config_from_properties()`: adaptive configuration generation

**Property system**:
- 5 property groups: analyzer, periodicity, decomposition, outlier, scaling
- JSON serialization for complex data structures  
- Automatic max_age management by data interval
- Force recalculate mechanism for mandatory updates

## 📁 File Structure

| File | Purpose | Key Components |
|------|---------|----------------|
| `pipeline.py` | Processing pipeline | ProcessingPipeline, sequential execution |
| `preprocessor.py` | System facade | TimeSeriesPreprocessor, pipeline coordination |
| `preprocessingConfig.py` | Property configuration | PropertyGroupConfig, classification functions |
| `propertyManager.py` | Persistence manager | PropertyManager, DB operations, caching |
| `analyzer/` | Stage 1: Statistical analysis | AnalysisProcessor, 20+ characteristics |
| `periodicity/` | Stage 2: Periodicity detection | PeriodicityDetectorProcessor, ensemble methods |
| `decomposition/` | Stage 3: Component decomposition | DecompositionProcessor, 7 algorithms |

## 🔄 Integration

**Input data**:
- `df`: pd.DataFrame with time series (timestamp + target columns)
- Configuration: ts_id, interval, instrument_type via constructor
- Optional: force_recalculate, custom configs for each stage

**Output data**:
- `X`: pd.DataFrame with features and decomposition components
- `y`: pd.Series with scaled target variable  
- Side effects: automatic property saving to DB when necessary

**Pipeline Flow**:
```
Raw TimeSeries → Analysis → Periodicity → Decomposition → [OutlierRemover] → [FeatureEngineer] → ML-ready data
```

**Dependencies**:
- `helpers/configs`: InstrumentTypeConfig, PropertySourceConfig
- `helpers/protocols`: TimeSeriesTransformProcessorProtocol
- `helpers/qualityEvaluator`: QualityEvaluator for decomposition
- External: pandas, numpy, scipy, statsmodels, sklearn

**Error Handling**:
- Graceful degradation: fallback to basic methods on errors
- Stage isolation: error in one stage doesn't stop pipeline
- Comprehensive logging: detailed diagnostics of all operations

## 🚀 Usage Examples

### Basic usage
```python
preprocessor = TimeSeriesPreprocessor(
    ts_id="CRYPTO_BTC_1h",
    db_session=db_session,
    targetColumn="close"
)

X, y = preprocessor.process(dataframe)
# X contains trend, seasonal, residual components + characteristics
# y contains scaled target variable
```

### Force recalculation of all stages  
```python
preprocessor = TimeSeriesPreprocessor(
    ts_id="CRYPTO_ETH_1d",
    db_session=db_session,
    forceRecalculate=True  # or {"analyzer": True, "periodicity": False}
)

X, y = preprocessor.process(dataframe)
```

### Custom stage configuration
```python
# Through PropertyManager you can get/modify properties before processing
prop_manager = PropertyManager(interval="1h", db_session=db_session)
existing_props = prop_manager.get_all_properties("CRYPTO_BTC_1h")

# Modification and reprocessing
preprocessor = TimeSeriesPreprocessor(...)
```

## 📊 Property Groups and Their Characteristics

### ANALYZER (30+ fields)
- **Statistics**: volatility, skewness, kurtosis, lag1_autocorrelation
- **Stationarity**: is_stationary, adf_pvalue, kpss_pvalue  
- **Quality**: data_quality_score, missing_ratio, noise_level
- **Catch22**: 22 specialized time features

### PERIODICITY (10+ fields)  
- **Main**: main_period, periods, confidence_scores
- **Methods**: detection_method, method_results, detection_status
- **Context**: suggested_periods, quality_score

### DECOMPOSITION (35+ fields)
- **Components**: trend_strength, seasonal_strength, residual_strength  
- **Quality**: quality_score, reconstruction_error, baseline_quality
- **Method-specific**: ssa_components, fourier_harmonics, nbeats_architecture

### Future groups
- **OUTLIER**: contamination parameters, detection methods
- **SCALING**: scaler configs, normalization parameters

## ⚙️ Configuration System

### Automatic adaptation
**By data frequency**:
- HIGH (≤15m): increased windows, strict thresholds  
- MEDIUM (30m-12h): balanced parameters
- LOW (≥1d): compact windows, soft criteria

**By data length**:
- TINY (<50): only basic methods, minimal parameters
- LARGE (1000+): all algorithms, maximum accuracy
- MASSIVE (50000+): performance optimization

**By instrument type**:
- CRYPTO: threshold corrections for high volatility
- Special thresholds and adaptive criteria

### Property Management
**Max Age by intervals**:
- 1h data: analyzer=7d, periodicity=30d, decomposition=14d
- 1d data: analyzer=30d, periodicity=90d, decomposition=30d
- Automatic property staleness determination

## 🔮 Future Development

### Planned modules (after decomposition):
4. **OutlierRemover**: intelligent anomaly removal with decomposition awareness
5. **FeatureEngineer**: ML feature creation from components and characteristics

### Extended architecture:
```
Analysis → Periodicity → Decomposition → OutlierRemoval → FeatureEngineering → ML Pipeline
```

**Additional capabilities**:
- Streaming processing for real-time data
- Distributed processing for large datasets  
- Advanced ensemble methods for decomposition
- AutoML integration for automatic algorithm selection

---

> **Result**: raw time series is transformed into ML-ready dataset with complete set of components, characteristics, and metadata for high-precision analysis and forecasting.