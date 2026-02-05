# Periodicity Module

> Automatic detection of periodic patterns in financial time series through ensemble of detection methods

## 🎯 Purpose

The periodicity module performs intelligent periodicity detection in time series as the second pipeline stage after analyzer. It uses an ensemble of three mathematical methods (ACF, Spectral, Wavelet) with consensus voting for maximum detection accuracy. Adaptively configures itself based on data characteristics and instrument type.

Main task: detect main_period and additional periods with confidence assessment, which are then used by the decomposition module for configuring seasonal components and selecting appropriate decomposition method.

## 🔬 Mathematical Foundation

### ACF Method (Autocorrelation Function)
- **Foundation**: `γ(h) = Cov(X_t, X_{t+h}) / Var(X_t)` - series correlation with itself
- **Peak detection**: local maxima search with prominence filtering
- **Bias correction**: mathematically correct bias correction
- **Application**: stationary series with regular periodicity

### Spectral Method (Spectral Analysis)
- **Foundation**: Fast Fourier Transform (FFT) for frequency spectrum analysis
- **Welch method**: improved power spectral density estimation with overlap
- **Peak detection**: dominant frequency search with adaptive thresholds
- **Application**: high-frequency data with clear harmonics

### Wavelet Method (Wavelet Analysis)
- **Foundation**: Continuous Wavelet Transform (CWT) - time-frequency analysis
- **Scalogram**: wavelet coefficient power analysis
- **Ridge detection**: stable frequency component search
- **Application**: non-stationary and noisy data

### Consensus Ensemble
- **Weighted voting**: combining results with confidence-based weights
- **Period grouping**: grouping close periods with adaptive thresholds
- **Cross-validation**: statistical validation of found periods
- **Quality scoring**: comprehensive reliability assessment of detection

## 🏗️ Technical Implementation

**Architectural Pattern**: BaseProcessor Template Method + Strategy Pattern + Ensemble + Configuration Adapter Protocol

**Main Classes**:
- `PeriodicityDetectorProcessor`: inherits from `BaseProcessor`, pipeline integration with 84% code reduction
- `PeriodicityDetector`: inherits from `BaseAlgorithm`, method orchestrator with consensus voting
- `BasePeriodicityMethod`: inherits from `BaseTimeSeriesMethod`, base class for detection methods
- `PeriodicityConfigAdapter`: inherits from `BaseConfigAdapter`, adaptive configuration with 50% code reduction
- `ACFMethod`, `SpectralMethod`, `WaveletMethod`: concrete algorithms

**Key Methods**:
- `PeriodicityDetectorProcessor.process()`: main entry point from pipeline
- `PeriodicityDetector.process()`: ensemble method coordination
- `build_config_from_properties()`: adaptive configuration (1452 lines)

**Configuration System**:
- Implements ConfigurationAdapterProtocol
- BASE configurations for each method (ACF/Spectral/Wavelet)
- RULES adaptation by frequency/length/instrument_type
- Mathematical constraints (ACF lags ≤ length/3, FFT nperseg ≤ length/2)
- Cryptocurrency adjustments for high volatility

## 📁 File Structure

| File | Purpose | Key Components |
|------|---------|----------------|
| `__init__.py` | Export main classes | PeriodicityDetectorProcessor, PeriodicityDetector, config exports |
| `processorPeriodicityDetector.py` | Pipeline integration and lifecycle | PeriodicityDetectorProcessor, fallback logic, crypto adjustments |
| `algorithmPeriodicityDetector.py` | Detection method orchestrator | PeriodicityDetector, ensemble coordination, consensus voting |
| `configPeriodicity.py` | Adaptive configuration (1452 lines) | build_config_from_properties, crypto volatility thresholds |
| `methods/__init__.py` | Detection method registry | AVAILABLE_METHODS with metadata |
| `methods/basePeriodicityMethod.py` | Method base class | BasePeriodicityMethod, validation utilities |
| `methods/acfMethod.py` | Autocorrelation analysis | ACFMethod, FFT optimization, bias correction |
| `methods/spectralMethod.py` | Spectral analysis | SpectralMethod, Welch method, harmonic detection |
| `methods/waveletMethod.py` | Wavelet analysis | WaveletMethod, CWT, scalogram analysis |

## 🔄 Integration

**Input Data**:
- `data`: pd.DataFrame with time series (usually after analyzer)
- `context`: Dict with analyzer properties (volatility, stationarity, noise_level)
- Optional: `properties` (from DB), `detectorConfig` (custom configuration)

**Output Data**:
- `data`: DataFrame unchanged (passthrough)
- `context.currentProperties.periodicity`: Dict with detection results:
  - **Main**: main_period (int), periods (List[int]), confidence_scores (List[float])
  - **Metadata**: detection_method, detection_status, periodicity_quality_score
  - **Details**: method_results (JSON), acf_values, suggested_periods (heuristics)
  - **Consensus**: consensus_strength, validation_score, total_candidates

**Dependencies**:
- `baseModule/baseProcessor`: BaseProcessor for Template Method architecture
- `baseModule/baseAlgorithm`: BaseAlgorithm for unified ensemble workflow
- `baseModule/baseMethod`: BaseTimeSeriesMethod for method base functionality
- `helpers/configs`: InstrumentTypeConfig, PropertySourceConfig
- `helpers/protocols`: TimeSeriesTransformProcessorProtocol
- `timeSeriesProcessing/preprocessingConfig`: frequency/length classification
- External: scipy, numpy, statsmodels, PyWavelets

**Fallback Behavior**:
- On errors: heuristic periods by interval (`_get_heuristic_periods`) through BaseProcessor fallback mechanism (SB8-68)
- Graceful degradation: main periods for cryptocurrencies (24h, 168h for daily)
- Template Method handling: unified through BaseProcessor._handle_processor_error()
- Error context: detailed diagnostics in context.error

## 🚀 Usage Examples

### Basic Pipeline Usage
```python
processor = PeriodicityDetectorProcessor(
    ts_id="BTC_USD",
    currency="USD",
    interval="1h", 
    instrument_type=InstrumentTypeConfig.CRYPTO,
    targetColumn="close"
)

data, context = processor.process(dataframe, context)
period_props = context["currentProperties"]["periodicity"]
main_period = period_props["main_period"]  # e.g., 24 for daily cycle
```

### Custom Configuration
```python
custom_config = {
    "_active_methods": ["acf", "spectral"],
    "acf": {"correlation_threshold": 0.3, "use_fft": True},
    "spectral": {"n_peaks": 5, "use_welch": True}
}

processor = PeriodicityDetectorProcessor(..., detectorConfig=custom_config)
```

### Direct Detector Usage
```python
detector = PeriodicityDetector(config)
result = detector.process(series, context)
```

## 📊 Adaptive Configuration

System automatically adjusts parameters for:

**Instrument Type**: 
- CRYPTO: threshold adjustments for high periodicity volatility
- Special thresholds: stable (10%), variable (30%), chaotic (50%+)

**Data Frequency**:
- HIGH (≤5m): more lags, strict thresholds, increased windows
- MEDIUM (5m-1h): balanced parameters
- LOW (≥1h): fewer lags, soft thresholds, compact windows

**Data Length**:
- TINY (<50): ACF only, minimal parameters
- SHORT (50-200): ACF+Spectral, base settings
- LARGE (1000+): all methods, maximum precision

**Mathematical Constraints**:
- ACF max_lags constrained by data length (≤ length/3)
- Spectral nperseg automatically adapts (8-512)
- Wavelet n_scales accounts for cone of influence (≤ length/4)

## 🎯 Detection Status Values

System evaluates detection quality:
- **"high_confidence"**: quality_score ≥ 0.75, strong consensus
- **"detected"**: quality_score ≥ 0.55, moderate confidence
- **"low_confidence"**: quality_score ≥ 0.35, weak signals
- **"spurious"**: quality_score < 0.35, likely false positive
- **"not_detected"**: main_period = 0, no significant periods

## 🔧 Heuristic Periods

Each interval has empirical periods defined:
- **1h**: [24, 168] (day, week)
- **1d**: [7, 30] (week, month)  
- **1m**: [60, 1440] (hour, day)
- Used as fallback and for validation

---

> **Result**: time series receives accurate periodicity assessment with method consensus for reliable decomposition strategy selection in downstream modules.