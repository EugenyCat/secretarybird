# Decomposition Module

> Intelligent time series decomposition with ensemble of 7 algorithms and mathematically optimized adaptive configuration

## 🎯 Purpose

The decomposition module performs intelligent decomposition of time series into trend, seasonal and residual components as the third stage of the pipeline after analyzer and periodicity. Uses enhanced decision tree for automatic selection of optimal algorithm from 7 methods based on data characteristics and quality assessment of results.

Main task: decompose time series into interpretable components (trend + seasonality + residuals) with maximum quality for subsequent machine learning or analytics stages. The system automatically selects method and its parameters for data specifics.

## 🔬 Mathematical Foundation

### Enhanced Decision Tree (Intelligent Algorithm Selection)
Multi-level data classification system for selecting optimal method:
- **Data characteristics**: length, stationarity, noise_level, volatility, trend_strength
- **Periodicity context**: main_period, seasonal_strength from previous stage
- **Quality optimization**: iterative selection through QualityEvaluator with ensemble scoring

### 7 Decomposition Algorithms

**Fourier Decomposition** - O(n log n)
- **Application**: stationary series with clear periodicity
- **Mathematics**: FFT for extracting harmonic components, Nyquist frequency analysis
- **Features**: optimal for regular seasonality, high performance

**SSA (Singular Spectrum Analysis)** 
- **Application**: nonstationary series with high autocorrelation
- **Mathematics**: SVD decomposition of trajectory matrix, signal subspace separation
- **Features**: model-free approach, good for weak periodicity

**TBATS (Trigonometric Box-Cox ARMA Trend Seasonal)**
- **Application**: multiple seasonality with Box-Cox transformation  
- **Mathematics**: Box-Cox + trigonometric seasonal + ARMA(p,q) errors
- **Features**: handles complex seasonal patterns, non-Gaussian distributions

**MSTL (Multiple Seasonal Trend decomposition using Loess)**
- **Application**: multiple seasonal patterns
- **Mathematics**: nested STL with different seasonal windows
- **Features**: robustness to outliers, flexible seasonal modeling

**RobustSTL** 
- **Application**: noisy data with outliers
- **Mathematics**: weighted STL with outlier-resistant components
- **Features**: high resistance to anomalies

**Prophet (Facebook Prophet)**
- **Application**: trends with holidays and structural breaks
- **Mathematics**: additive/multiplicative components with changepoint detection
- **Features**: interpretable trends, holiday effects

**N-BEATS (Neural Basis Expansion Analysis)**
- **Application**: neural decomposition with harmonic blocks
- **Mathematics**: deep learning with interpretable trend/seasonal blocks
- **Features**: cutting-edge approach, high accuracy for complex patterns

### Mathematically Optimized Configuration (2951 lines)
- **Cleveland et al.** formulas for STL seasonal windows
- **Golyandina theory** for SSA window optimization
- **Nyquist analysis** for Fourier harmonics 
- **Exponential adaptation** for Prophet scaling parameters

## 🏗️ Technical Implementation

**Architectural pattern**: BaseProcessor Template Method + Strategy Pattern + Enhanced Decision Tree + Quality-based Selection

**Main classes**:
- `DecompositionProcessor`: inherits from `BaseProcessor`, pipeline integration with 86% code reduction
- `DecompositionAlgorithm`: inherits from `BaseAlgorithm`, method orchestrator with enhanced decision tree
- `BaseDecomposerMethod`: inherits from `BaseTimeSeriesMethod`, base class for all decomposition methods
- `DecompositionConfigAdapter`: inherits from `BaseConfigAdapter`, adaptive configuration with 59% code reduction
- Concrete methods: `FourierDecomposerMethod`, `SSADecomposerMethod`, etc.

**Key methods**:
- `DecompositionProcessor.process()`: main entry point from pipeline
- `DecompositionAlgorithm.process()`: enhanced decision tree + execution
- `build_config_from_properties()`: mathematically optimized adaptation
- `QualityEvaluator.evaluate_decomposition()`: results quality assessment

**Configuration system** (V5.0.0):
- Implements ConfigurationAdapterProtocol  
- CRYPTO_HIGH_VOLATILITY_ADJUSTMENTS for cryptocurrencies
- Mathematically correct formulas from scientific literature
- Full validation of all parameters with theoretical constraints

## 📁 File Structure

| File | Purpose | Key Components |
|------|---------|----------------|
| `processorDecomposition.py` | Pipeline integration and lifecycle | DecompositionProcessor, fallback logic, error handling |
| `algorithmDecomposition.py` | Method orchestrator and decision tree | DecompositionAlgorithm, enhanced method selection |
| `configDecomposition.py` | Mathematically optimized configuration (2951 lines) | build_config_from_properties, crypto adaptations |
| `methods/baseDecomposerMethod.py` | Base class for methods | BaseDecomposerMethod, quality assessment |
| `methods/fourierDecomposerMethod.py` | Fourier decomposition | FFT-based decomposition, harmonic analysis |
| `methods/ssaDecomposerMethod.py` | Singular Spectrum Analysis | SVD decomposition, trajectory matrix |
| `methods/tbatsDecomposerMethod.py` | TBATS algorithm | Box-Cox + trigonometric + ARMA |
| `methods/mstlDecomposerMethod.py` | Multiple STL | Nested Loess decomposition |
| `methods/robustSTLDecomposerMethod.py` | Robust STL | Outlier-resistant decomposition |
| `methods/prophetDecomposerMethod.py` | Facebook Prophet | Trend changepoints, holiday effects |
| `methods/nbeatsDecomposerMethod.py` | N-BEATS neural decomposition | Deep learning, interpretable blocks |

## 🔄 Integration

**Input data**:
- `data`: pd.DataFrame with time series after analyzer/periodicity stages
- `context`: Dict with analyzer/periodicity properties (volatility, main_period, etc.)
- Configuration: automatic adaptation or custom decompositionConfig

**Output data**:
- `data`: DataFrame with added components:
  - `{targetColumn}_trend`: trend component
  - `{targetColumn}_seasonal`: seasonal component  
  - `{targetColumn}_residual`: residual component
  - `{targetColumn}_scaled`: scaled target variable
- `context.currentProperties.decomposition`: Dict with 35+ quality characteristics

**Used Helpers**:

### BoxCoxTransformer (pipeline/helpers/boxCoxTransformer.py)
- **Purpose**: numerically stable Box-Cox transformations for TBATS and other methods
- **Functions**: automatic lambda optimization, overflow protection, chunked processing
- **Integration**: automatic application when variance stabilization needed

### QualityEvaluator (pipeline/helpers/qualityEvaluator.py)  
- **Purpose**: universal decomposition quality assessment system
- **Metrics**: MSE/MAE, AIC/BIC, seasonal/trend strength, residual autocorrelation, robustness
- **Integration**: composite scoring for selecting best method in enhanced decision tree

**Dependencies**:
- `baseModule/baseProcessor`: BaseProcessor for Template Method architecture
- `baseModule/baseAlgorithm`: BaseAlgorithm for unified decision tree workflow
- `baseModule/baseMethod`: BaseTimeSeriesMethod for base method functionality
- `helpers/qualityEvaluator`: QualityEvaluator, QualityMetricConfig
- `helpers/boxCoxTransformer`: BoxCoxTransformer for variance stabilization
- `helpers/protocols`: TimeSeriesTransformProcessorProtocol
- External: scipy, numpy, statsmodels, sklearn, pytorch (for N-BEATS)

## 🚀 Usage Examples

### Basic pipeline usage
```python
processor = DecompositionProcessor(
    ts_id="CRYPTO_BTC_1h",
    currency="USD",
    interval="1h", 
    instrument_type=InstrumentTypeConfig.CRYPTO,
    targetColumn="close"
)

data, context = processor.process(dataframe, context)
decomp_props = context["currentProperties"]["decomposition"]

# Access components
trend = data["close_trend"]
seasonal = data["close_seasonal"] 
residual = data["close_residual"]
scaled_target = data["close_scaled"]
```

### Custom method configuration
```python
custom_config = {
    "_active_methods": ["fourier", "ssa", "tbats"],
    "fourier": {"n_harmonics": 10, "detrend": True},
    "ssa": {"window_length": 144, "n_components": 20}
}

processor = DecompositionProcessor(..., decompositionConfig=custom_config)
```

### Direct algorithm usage
```python
algorithm = DecompositionAlgorithm(config)
result = algorithm.process(series, context)
```

## 📊 Enhanced Decision Tree Logic

### Data Characteristics Classification
**Noise Level Categories**:
- ultra_low (≤0.5%), very_low (≤1%), low (≤5%), medium (≤10%), high (≤20%), extreme (>35%)

**Trend Strength Categories**: 
- none (≤1%), weak (≤5%), moderate (≤15%), strong (≤30%), dominant (≤50%), overwhelming (>70%)

**Crypto Volatility Categories**:
- ultra_stable (≤2%), stable (≤10%), normal (≤20%), variable (≤30%), high (≤40%), extreme (>50%)

### Automatic Method Selection
- **High periodicity + low noise**: Fourier (optimal performance)
- **Complex seasonality**: MSTL or TBATS (multiple periods)
- **High noise + outliers**: RobustSTL (resistance to anomalies)
- **Nonstationarity**: SSA (model-free approach)
- **Structural breaks**: Prophet (changepoint detection)
- **Complex patterns**: N-BEATS (deep learning power)

### Quality-based optimization through BaseAlgorithm
- Enhanced Decision Tree integrated into BaseAlgorithm Template Method (SB8-67)
- Iterative method testing with QualityEvaluator scoring through `_combine_results()` strategy
- Composite metrics: reconstruction error + component strength + robustness
- Fallback chain on errors: complex → simple → baseline methods through BaseProcessor error handling

## ⚙️ Configuration Adaptation

### Mathematically Optimized Formulas
- **STL seasonal window**: Cleveland formula with adaptation for data length
- **SSA window length**: Golyandina optimal range (N/3 for trajectory matrix)
- **Fourier harmonics**: Nyquist-based calculation with noise filtering
- **Prophet seasonality**: exponential scaling for data frequency

### Crypto-specific Adjustments  
- **High volatility adjustments**: parameter softening for stability
- **Outlier tolerance**: increased thresholds for crypto market dynamics
- **Seasonal detection**: adaptive criteria for 24/7 trading

### Automatic parameter bounds
- All parameters have theoretically grounded constraints
- Mutual coprimality of periods for multiple seasonality
- Memory optimization for large datasets

## 🎯 Quality and Metrics

### Decomposition Quality Assessment
- **MSE/MAE**: reconstruction error between original and sum of components
- **Seasonal/Trend Strength**: Hyndman formulas for component assessment  
- **Residual Analysis**: autocorrelation tests, normality checks
- **Robustness**: MAD vs STD ratios, outlier sensitivity

### Output Properties (35+ fields)
- **Basic**: decomposition_method, quality_score, reconstruction_error
- **Components**: trend_strength, seasonal_strength, residual_strength
- **Method-specific**: ssa_window_length, fourier_n_harmonics, nbeats_architecture_efficiency
- **Quality metrics**: baseline_quality, stability_metrics, corrections_applied

---

> **Result**: time series is decomposed into interpretable components with maximum quality through automatic selection of optimal algorithm and mathematically grounded parameter configuration.