# Analyzer Module

> Comprehensive time series analysis through a system of modular methods with adaptive configuration

## 🎯 Purpose

The analyzer module performs multi-level time series analysis as the first stage in the preprocessing pipeline. Creates basic characteristics and data properties for subsequent modules (decomposition, periodicity). Uses adaptive configuration that adjusts to instrument type, data frequency, and time series length.

Main task: transform raw time series into a structured set of properties (analyzer properties), which is then used for decision-making in other pipeline modules.

## 🔬 Mathematical Foundation

### Statistical Analysis
- **Basic statistics**: mean, std, min/max, skewness, kurtosis, quartiles (Q1/Q3/IQR)
- **Volatility metrics**: coefficient of variation, returns volatility 
- **Trend analysis**: trend strength estimation through decomposition into trend/noise
- **Missing data**: missing ratio, missing values count
- **Catch22 features**: 22 specialized temporal features (mode, periodicity, entropy, etc.)

### Stationarity Tests
- **ADF test**: Augmented Dickey-Fuller with configurable alpha
- **KPSS test**: Kwiatkowski-Phillips-Schmidt-Shin with adaptive thresholds
- **Rolling stability**: CV rolling mean/std to check parameter stability
- **Consensus approach**: combining results from multiple tests

### Outlier Detection
- **Z-score method**: `|z| > threshold` with adaptive thresholds  
- **IQR method**: `Q1 - k*IQR` and `Q3 + k*IQR` boundaries
- **MAD method**: Median Absolute Deviation with robust estimation
- **Adaptive thresholds**: threshold adjustment for cryptocurrencies (high volatility)

### Autocorrelation Analysis
- **Lag-1 autocorrelation**: primary autocorrelation for AR(1) check
- **Significance testing**: statistical significance of autocorrelations
- **Extended lags**: analysis up to `autocorr_max_lag` with significant lag filtering

## 🏗️ Technical Implementation

**Architectural pattern**: BaseProcessor Template Method + Strategy Pattern + Configuration Adapter Protocol

**Main classes**:
- `AnalysisProcessor`: inherits from `BaseProcessor`, pipeline integration with 85% code reduction
- `TimeSeriesAnalyzer`: inherits from `BaseAlgorithm`, method orchestrator with ensemble execution
- `BaseAnalysisMethod`: inherits from `BaseTimeSeriesMethod`, base class for all analysis methods
- `AnalyzerConfigAdapter`: inherits from `BaseConfigAdapter`, adaptive configuration with 62% code reduction
- `StatisticalMethod`, `StationarityMethod`, `OutlierAnalysisMethod`: concrete methods

**Key methods**:
- `AnalysisProcessor.process()`: main entry point from pipeline
- `TimeSeriesAnalyzer.process()`: method execution coordination  
- `TimeSeriesAnalyzer.process_with_dataframe_enrichment()`: DataFrame enrichment
- `OutlierAnalysisMethod.process_with_dataframe_enrichment()`: outlier boolean columns
- `build_config_from_properties()`: adaptive configuration for data type

**Configuration system**:
- BASE configurations for each method
- RULES adaptation by frequency/length/instrument type  
- Range validation and data constraints
- Special adjustments for cryptocurrencies

## 📁 File Structure

| File | Purpose | Key Components |
|------|---------|----------------|
| `__init__.py` | Export main classes | AnalysisProcessor, TimeSeriesAnalyzer |
| `processorAnalyzer.py` | Pipeline integration and lifecycle | AnalysisProcessor, fallback logic, DataFrame enrichment |
| `algorithmAnalyzer.py` | Analysis method orchestration | TimeSeriesAnalyzer, ensemble execution, **process_with_dataframe_enrichment()** |
| `configAnalyzer.py` | Adaptive configuration (930 lines) | build_config_from_properties, crypto adaptations, validation |
| `methods/__init__.py` | Analysis method registry | METHOD_REGISTRY, export all methods |
| `methods/baseAnalysisMethod.py` | Base method class | BaseAnalysisMethod, common interface |
| `methods/statisticalMethod.py` | Statistical analysis | StatisticalMethod, catch22, basic stats |
| `methods/stationarityMethod.py` | Stationarity tests | StationarityMethod, ADF/KPSS tests |
| `methods/outlierAnalysisMethod.py` | Outlier detection | OutlierAnalysisMethod, Z-score/IQR/MAD, **DataFrame enrichment** |

## 🔄 Integration

**Input data**:
- `data`: pd.DataFrame with time series (one column)
- `context`: Dict with metadata (interval, instrument_type, targetColumn)
- Optional: `properties` (existing from database), `analyzerConfig` (override)

**Output data**:
- `data`: DataFrame with added boolean outlier columns:
  - `is_zscore_outlier`: boolean markers for Z-score outliers
  - `is_iqr_outlier`: boolean markers for IQR outliers  
  - `is_mad_outlier`: boolean markers for MAD outliers
- `context.currentProperties.analyzer`: Dict with 20+ characteristics:
  - Basic: length, missing_ratio, volatility, skewness, kurtosis
  - Outliers: zscore_outliers, iqr_outliers, mad_outliers, outlier_ratio
  - Stationarity: is_stationary, adf_pvalue, kpss_pvalue
  - Quality: data_quality_score, noise_level, trend_strength
  - Catch22: c22_mode_5, c22_periodicity, c22_entropy_pairs, etc.

**Dependencies**:
- `baseModule/baseProcessor`: BaseProcessor for Template Method architecture
- `baseModule/baseAlgorithm`: BaseAlgorithm for unified workflow
- `baseModule/baseMethod`: BaseTimeSeriesMethod for base method functionality
- `helpers/configs`: InstrumentTypeConfig, PropertySourceConfig
- `helpers/protocols`: TimeSeriesTransformProcessorProtocol
- `helpers/utils`: mathematical functions, validation
- `timeSeriesProcessing/preprocessingConfig`: data classification

**Fallback behavior**:
- On errors: transition to simplified analysis (fallbackBehavior='simple')
- Graceful degradation: return basic statistics on failures
- Error context: detailed error information in context.error

## 🔄 Repeated Run Logic

**Initial calculation** (`properties = None`):
- Full time series analysis through all active methods
- Calculation of all statistical characteristics (analyzer properties)
- DataFrame enrichment with addition of outlier boolean columns
- Save properties to database with PropertySourceConfig.CALCULATED

**Repeated calculation** (existing `properties`):
- Reuse properties from database (without recalculation of analysis)
- Only DataFrame enrichment for consistent output format
- 10x performance improvement on repeated runs
- PropertySourceConfig.DATABASE for tracking data source

**Architectural consistency**: Pattern matches decomposition module with similar component reuse logic.

## 🚀 Usage Examples

### Basic usage in pipeline
```python
processor = AnalysisProcessor(
    ts_id="BTC_USD",
    currency="USD", 
    interval="1h",
    instrument_type=InstrumentTypeConfig.CRYPTO,
    targetColumn="close"
)

data, context = processor.process(dataframe, context)
properties = context["currentProperties"]["analyzer"]
```

### Custom configuration
```python
custom_config = {
    "_active_methods": ["statistical", "outlier"],
    "statistical": {"calculate_advanced": True, "autocorr_max_lag": 100},
    "outlier": {"zscore_threshold": 2.5}
}

processor = AnalysisProcessor(..., analyzerConfig=custom_config)
```

### Direct analyzer usage
```python
analyzer = TimeSeriesAnalyzer(config)
result = analyzer.process(series, context)
```

## 📊 Adaptive Configuration

The system automatically adjusts parameters for:

**Instrument type**: CRYPTO gets special adjustments for high volatility
**Data frequency**: HIGH/MEDIUM/LOW affects window sizes and thresholds
**Data length**: TINY/SHORT/SMALL/LARGE/HUGE adjusts lags and minimum requirements

Example for CRYPTO with high volatility:
- Outlier thresholds increase by 1.3-1.4x
- Stationarity tests become less strict  
- Window sizes adapt to data frequency
- Catch22 features are enabled only with sufficient data volume

---

> **Result**: time series receives a complete set of characteristics for decision-making in downstream modules, accounting for instrument and data specificity.