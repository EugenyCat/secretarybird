# Outlier Detection Module

> Production-ready outlier detection and removal with Enhanced Decision Tree, tiered execution, and 95%+ context reuse

## 🎯 Purpose

Performs intelligent outlier detection and removal in time series through a multi-tier method system with adaptive execution strategy. Uses Enhanced Decision Tree for method selection based on context characteristics (decomposition quality, volatility, market regime). Critical pipeline stage between decomposition and final processing, creates robust version of time series for subsequent analysis.

**Key Features**: Zero-cost detection through context reuse (95%+), early stopping optimization (0ms → 100ms), weighted consensus mechanism, regime-aware method selection, financial helpers integration for regime classification and HFT noise detection.

## 🔬 Mathematical Foundation

### Tier 1: Statistical Enhancement (Zero-cost)
- **Weighted majority voting**: consensus from boolean columns (Z-score, IQR, MAD)
- **Consensus strength**: `S(x) = Σ(w_i · I_i(x)) / Σ(w_i)` where `w_i` - method weights (MAD=0.9, IQR=0.8, Z=0.7)
- **Quality-weighted scaling**: `S_adj = S · (α + (1-α)·Q)` - linear shrinkage with quality prior
- **Reference**: Kuncheva (2014) "Combining Pattern Classifiers", Rousseeuw & Hubert (2011)

### Tier 2: Component Anomaly (Decomposition-based)
- **Trend anomalies**: Percentile-based changepoint detection `|Δtrend| > P_95(|Δtrend|)`
- **Seasonal/Residual anomalies**: MAD-based robust z-score `|x - median| / MAD > k`, where `k ∈ [2.5, 6.0]`
- **Kurtosis adjustment**: `MAD_constant = 3.5 · max(1, kurtosis/3.5)` for heavy-tailed distributions
- **Weighted combination**: `outlier(x) = Σ(strength_i · anomaly_i(x)) > threshold`
- **Reference**: Iglewicz & Hoaglin (1993), Leys et al. (2013), Cleveland et al. (1990)

### Enhanced Decision Tree Logic
```
IF decomposition.quality_score > tier2_quality_threshold (default: 0.5)
  THEN execute Tier 1 + Tier 2
  ELSE execute Tier 1 only (early stopping)

IF Tier1.mean_consensus_strength > early_stopping_threshold (default: 0.95)
  THEN stop (sufficient confidence)
  ELSE continue to Tier 2
```

### Adaptive Weighting Strategy (v2.0)

**NEW in v2.0 (SB8-104)**: Dynamic tier weights based on decomposition quality instead of fixed regime weights.

OutlierDetection automatically adjusts method combination weights based on `quality_score` and `residual_strength` from decomposition:

| quality_score | residual_strength | T1 (Statistical) | T3 (Component) | Strategy |
|---------------|-------------------|------------------|----------------|----------|
| **HIGH** (≥0.7) | ≥0.05 (strong) | 30% | 70% | **Trust components** - excellent decomposition |
| **MEDIUM** (0.5-0.7) | ≥0.02 (normal) | 50% | 50% | **Balanced** - acceptable quality |
| **MEDIUM** (0.5-0.7) | <0.02 (weak) | 60% | 40% | **Conservative** - weak residual signal |
| **CRITICAL** (<0.5) | any | 100% | 0% | **Fallback** - disable components (poor quality) |

**Quality-based protection**:
- `min_quality_threshold = 0.5` (production safety)
- Component detection **DISABLED** when `quality_score < 0.5` to prevent false positives
- Quality warnings generated: CRITICAL (<0.5), WARNING (0.5-0.7)
- Graceful degradation to raw price statistics only

**Example flow**:
```python
# High quality decomposition (0.85) + strong residual (0.08)
weights = {"statistical_enhancement": 0.3, "component_anomaly": 0.7}
# → Trust component-based detection

# Poor decomposition (0.3)
weights = {"statistical_enhancement": 1.0, "component_anomaly": 0.0}
quality_warnings = ["CRITICAL: Poor decomposition quality. Component detection DISABLED."]
# → Fallback to raw price statistics only
```

**Configuration**:
```python
"adaptive_weighting": {
    "enabled": True,
    "high_quality_threshold": 0.7,
    "medium_quality_threshold": 0.5,
    "min_quality_threshold": 0.5,
    "strong_residual_threshold": 0.05,
    "weak_residual_threshold": 0.02
}
```

### Regime-Aware Weighting (deprecated in v2.0)
- **Volatile Trending**: Component-heavy (40/60) - structural changes important
- **Volatile Ranging**: Balanced (50/50)
- **Calm Trending**: Component-heavy (30/70) - trend components dominant
- **Calm Ranging**: Statistical-heavy (60/40) - stable patterns
- **Edge cases**: Perfect persistence/reversion → equal weights (50/50)

### Consensus Mechanism
- **Binary classification**: `outlier = (weighted_consensus > 0.5)`
- **Confidence score**: normalized weighted sum `conf(x) = Σ(w_method · S_method(x)) / Σ(w_method)`
- **Outlier types**: statistical (Tier 1 only), component (Tier 2 only), consensus (both), none

## 🏗️ Technical Implementation

**Architectural Pattern**: Processor + Algorithm + Methods (BaseProcessor Template Method v2.0.0)

**Main Classes**:
- `OutlierDetectionProcessor` (Level 1) - BaseProcessor inheritance, DataFrame enrichment
- `OutlierDetectionAlgorithm` (Level 2) - Enhanced Decision Tree, tiered execution
- `OutlierDetectionConfigAdapter` - BASE/ACTIVE/RULES adaptive configuration
- `BaseOutlierDetectionMethod` - base class for detection methods
- `StatisticalEnhancementMethod` (Tier 1) - zero-cost consensus
- `ComponentAnomalyMethod` (Tier 2) - decomposition-based detection

**Key Methods**:
- `process()` (Processor) - main entry point, 8-step workflow
- `_execute_algorithm()` (Processor) - algorithm orchestration
- `process()` (Algorithm) - tiered method execution, weighted consensus
- `_analyze_context_and_select_methods()` - Enhanced Decision Tree
- `detect()` (Methods) - specific detection implementation

## 📁 File Structure

| File/Folder | Purpose | Key Components |
|---|---|---|
| `processorOutlierDetection.py` | Level 1 Processor (682 lines) | OutlierDetectionProcessor, DataFrame enrichment (6 columns) |
| `algorithmOutlierDetection.py` | Level 2 Algorithm (1201 lines) | OutlierDetectionAlgorithm, Enhanced Decision Tree, weighted consensus |
| `configOutlierDetection.py` | Configuration Adapter (751 lines) | OutlierDetectionConfigAdapter, BASE/ACTIVE/RULES, mathematical constraints |
| `__init__.py` | Module exports | Architecture overview, key features |
| `methods/` | Detection methods (Tier 1-2) | — |
| `methods/baseOutlierDetectionMethod.py` | Base class for methods | BaseOutlierDetectionMethod, boolean columns utilities, consensus calculation |
| `methods/statisticalEnhancementMethod.py` | Tier 1: Zero-cost detection (514 lines) | StatisticalEnhancementMethod, weighted voting, quality scaling |
| `methods/componentAnomalyMethod.py` | Tier 2: Decomposition-based (825 lines) | ComponentAnomalyMethod, trend/seasonal/residual detection, kurtosis adjustment |

## 🔄 Integration

**Input Data**:
- `pd.DataFrame` with target column + decomposition components (trend, seasonal, residual)
- Boolean columns: `is_zscore_outlier`, `is_iqr_outlier`, `is_mad_outlier` (from Analyzer)
- `context["currentProperties"]`:
  - `analyzer`: volatility, data_quality_score, outlier_ratio, catch22_features
  - `decomposition`: quality_score, trend_strength, seasonal_strength, residual_strength

**Output Data**:
- Enriched DataFrame with 5 new columns:
  - `outliers`: Boolean mask consensus outliers
  - `outlier_confidence`: Float confidence scores [0, 1]
  - `outlier_score_enhanced`: Quality-weighted scores
  - `outlier_type`: Type classification (statistical/component/consensus/none)
  - `price_robust`: Interpolated robust price (outliers removed)
- `context["currentProperties"]["outlier_detection"]`:
  - `outliers_detected`, `outlier_ratio`, `quality_score`
  - `tier_reached`, `methods_used`, `regime`
  - `context_reuse_ratio`, `execution_time_ms`
  - `microstructure_analysis`: Dict with session-level HFT metrics
    - `hft_noise_level`: Float [0.0, 1.0] - HFT noise intensity
    - `hft_patterns`: List[str] - Detected HFT patterns
    - `applicable_for_data`: Bool - True if intraday data (microstructure applicable)
  - **NEW v2.0 fields** (SB8-104):
    - `adaptive_weights`: Dict - actual weights used for method combination
    - `decomposition_quality_score`: Float - quality from decomposition
    - `residual_strength`: Float - residual component strength
    - `quality_warnings`: List[str] - quality-based warnings (CRITICAL/WARNING)
    - `skipped_methods`: List[Dict] - methods skipped due to quality
  - `config_outlier_detection` (for state restoration)

**Dependencies**:
- **REQUIRED**: `analyzer` → `decomposition` → `outlier_detection` (order critical)
- **BaseProcessor**: `baseModule/baseProcessor.py` (template method pattern)
- **BaseMethod**: `baseModule/baseMethod.py` (method inheritance)
- **BaseConfigAdapter**: `baseModule/baseConfigAdapter.py` (configuration)
- **Financial Helpers** (optional):
  - `helpers/financial/regime.py` - classify_market_regime (comprehensive regime)
  - `helpers/financial/microstructure.py` - analyze_microstructure_patterns (HFT noise)
- **Utils**: `helpers/configs.py`, `helpers/utils.py`, `helpers/protocols.py`

## 🎨 Design Patterns

### 1. Enhanced Decision Tree
Adaptive method selection based on context characteristics:
```
Tier 1: ALWAYS execute (zero-cost)
Tier 2: IF decomposition.quality_score > 0.5 → execute
        IF Tier1.strength > 0.95 → early stop
```

### 2. Statistical Enhancement Pattern
95%+ context reuse through boolean columns:
- Zero computational cost (~0ms)
- Pure reuse of analyzer results
- Weighted consensus mechanism
- Quality-weighted scaling

### 3. Component-Aware Processing
Decomposition of components for anomaly detection:
- Trend changepoint detection (percentile-based)
- Seasonal MAD deviation (kurtosis-adjusted)
- **Residual consensus detection (v2.2.0 - SB8-103)**: Weighted voting of 3 methods
  - **MAD-based** (weight=0.5): Production-tested, 50% breakdown point, most robust
  - **Z-score** (weight=0.3): Assumes normality, medium robustness  
  - **IQR** (weight=0.2): Quantile-based, conservative, no distributional assumptions
  - **Consensus threshold**: 0.6 - outlier if `weighted_sum > threshold`
  - **Adaptive thresholds**: scale with residual strength for dynamic sensitivity
  - **Zero division protection**: handles constant/near-constant residuals gracefully
- Strength-proportional weighting

### 4. Tiered Execution Strategy
Early stopping optimization:
- Tier 1 (0ms): Statistical Enhancement
- Early stop check: if confidence > 0.95 → finish
- Tier 2 (+10ms): Component Anomaly (if quality sufficient)
- Weighted consensus: combine results

### 5. Regime-Aware Method Selection
Financial helpers integration for adaptive weighting:
- Classify market regime (volatile/calm × trending/ranging)
- Adjust method weights based on regime
- Detect HFT microstructure noise (session-level flags)
- Graceful fallback to simple heuristics

## ⚙️ Configuration Structure

### BASE Configuration
```python
BASE = {
    "base": {
        "min_data_length": 30,
        "consensus_threshold": 2,  # methods agreement
        "confidence_threshold": 0.5,
        # NEW v2.0: Algorithm-level parameters
        "early_stopping_threshold": 0.95,
        "tier2_decomposition_quality_threshold": 0.5,
        "enable_regime_classification": True
    },
    "statistical_enhancement": {
        "consensus_threshold": 2,
        "high_strength_threshold": 0.7,
        "method_weights": {"zscore": 0.7, "mad": 0.9, "iqr": 0.8},
        "quality_prior_weight": 0.5
    },
    "component_anomaly": {
        "trend_diff_percentile": 95,
        "seasonal_mad_threshold": 3.5,  # [2.5, 6.0]
        "residual_mad_threshold": 3.5,
        "anomaly_threshold": 0.5,
        # NEW v2.2.0: Residual consensus detection (SB8-103)
        "residual_method_weights": {
            "mad": 0.5,      # Highest (production-tested)
            "zscore": 0.3,   # Medium (normality assumption)
            "iqr": 0.2       # Lower (quantile-based)
        },
        "residual_consensus_threshold": 0.6,
        "residual_zscore_threshold": 3.5,
        "residual_iqr_multiplier": 1.5
    },
    # NEW v2.0: Adaptive Weighting (SB8-104)
    "adaptive_weighting": {
        "enabled": True,
        "high_quality_threshold": 0.7,
        "medium_quality_threshold": 0.5,
        "min_quality_threshold": 0.5,
        "strong_residual_threshold": 0.05,
        "weak_residual_threshold": 0.02
    }
}
```

### ACTIVE Methods
```python
ACTIVE = {
    InstrumentTypeConfig.CRYPTO: [
        "statistical_enhancement",  # Tier 1
        "component_anomaly"          # Tier 2
    ]
}
```

### RULES Adaptation
- **frequency**: HIGH/MEDIUM/LOW → adjust min_data_length, MAD thresholds
- **length**: TINY/SMALL/LARGE/HUGE/MASSIVE → adjust strength thresholds (inverted logic)
- **instrument**: CRYPTO → increase MAD thresholds ×1.3 (high volatility)

### Mathematical Constraints (v1.1.0)
- `mad_thresholds ∈ [2.5, 6.0]` (Iglewicz & Hoaglin 1993)
- `trend_diff_percentile ∈ [90, 99]` (Cleveland et al. 1990)
- `strength_thresholds ∈ [0, 1]` (probability bounds)
- `consensus_threshold ∈ [1, 3]` (method count)
- NaN-safe float conversions (v1.1.0 fix)

## 📊 Performance Characteristics

### Computational Cost
- **Tier 1 only**: ~0ms (boolean operations)
- **Tier 1 + Tier 2**: ~10-15ms (MAD + percentile calculations)
- **Early stopping**: 0ms → 100ms depending on confidence

### Context Reuse
- **Tier 1**: 95%+ (pure reuse of boolean columns)
- **Tier 2**: 95%+ (reuse of decomposition components)
- **Overall**: 95%+ context optimization

### Memory Footprint
- **Enrichment columns**: 6 × len(data) × 8 bytes (float64/bool)
- **Intermediate results**: minimal (lazy evaluation)
- **Config storage**: ~2-3 KB JSON

## 🔬 Production-Ready Features

### 1. Fail-Fast Validation
- Validate context dependencies (analyzer, decomposition)
- Validate boolean columns in DataFrame
- Validate decomposition component columns
- Fail-fast with clear error messages

### 2. Graceful Degradation
- Simple mode fallback (basic Z-score detection)
- Tier 2 skip if decomposition quality low
- Early stopping if Tier 1 confidence high
- Missing financial helpers → fallback heuristics

### 3. DataFrame Enrichment
5 new columns for downstream processing:
- `outliers`: consensus mask
- `outlier_confidence`: scores
- `outlier_score_enhanced`: quality-weighted
- `outlier_type`: classification
- `price_robust`: interpolated clean price

### 4. State Restoration
- `config_outlier_detection` saved in properties
- `_restore_module_state()` restores config + algorithm
- Lazy initialization for first run

### 5. Comprehensive Logging
- Input validation traces
- Method execution traces
- Tier reached logging
- Outlier counts and ratios
- Execution time metrics

## 📚 Version History

### v2.0.0 (2025-10-20) - Architectural Refactoring
- **REFACTORED**: Full BaseProcessor compliance
- **REMOVED**: `_validate_and_get_dataframe()` (HIGH-1 fix)
- **ADDED**: `_validate_context_dependencies()` (HIGH-5 fix)
- **FIXED**: Context access via `context["currentProperties"]` (HIGH-3)
- **ARCHITECTURAL**: Follows DecompositionProcessor pattern exactly
- **REDUCTION**: 85-86% code reduction through BaseProcessor inheritance

### v1.1.0 (2025-10-16) - Mathematical Validation
- **BLOCKING FIX**: MAD threshold bounds [2.5, 6.0] enforced
- **BLOCKING FIX**: Inverted data_length rules logic (×= 0.85 for LARGE)
- **HIGH FIX**: trend_diff_percentile bounds [90, 99]
- **HIGH FIX**: NaN-safe float conversions
- **APPROVED**: Mathematical validation by 5 independent experts

### v1.0.0 (2025-10-15) - Initial Production Release
- Enhanced Decision Tree implementation
- Tiered execution (Tier 1-2)
- Financial helpers integration
- Weighted consensus mechanism
- Production-ready framework

## 🔍 Mathematical Validation Status
**References**:
- Iglewicz & Hoaglin (1993): "How to Detect and Handle Outliers"
- Leys et al. (2013): "Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median"
- Cleveland et al. (1990): "STL: A Seasonal-Trend Decomposition Procedure Based on Loess"
- Rousseeuw & Hubert (2011): "Robust Statistics for Outlier Detection"
- Kuncheva (2014): "Combining Pattern Classifiers: Methods and Algorithms"
- Box & Jenkins (1976): "Time Series Analysis: Forecasting and Control"

## 💡 Usage Examples

### Basic Usage
```python
from pipeline.timeSeriesProcessing.outlierDetection import OutlierDetectionProcessor
from pipeline.helpers.configs import InstrumentTypeConfig

# Initialize processor
processor = OutlierDetectionProcessor(
    ts_id="BTC-USD",
    currency="BTC",
    interval="1h",
    instrument_type=InstrumentTypeConfig.CRYPTO,
    targetColumn="close"
)

# Process data (after analyzer and decomposition)
enriched_data, updated_context = processor.process(data, context)

# Access results
outliers_detected = updated_context["currentProperties"]["outlier_detection"]["outliers_detected"]
outlier_ratio = updated_context["currentProperties"]["outlier_detection"]["outlier_ratio"]
tier_reached = updated_context["currentProperties"]["outlier_detection"]["tier_reached"]

# Enriched DataFrame columns
outlier_mask = enriched_data["outliers"]
confidence_scores = enriched_data["outlier_confidence"]
robust_price = enriched_data["price_robust"]
```

### Configuration Override
```python
# Custom configuration
custom_config = {
    "_active_methods": ["statistical_enhancement", "component_anomaly"],
    "statistical_enhancement": {
        "consensus_threshold": 2,
        "high_strength_threshold": 0.8,
        "method_weights": {"zscore": 0.6, "mad": 1.0, "iqr": 0.7}
    },
    "component_anomaly": {
        "seasonal_mad_threshold": 4.0,
        "residual_mad_threshold": 4.0,
        "anomaly_threshold": 0.6
    },
    "early_stopping_threshold": 0.9
}

processor = OutlierDetectionProcessor(
    ts_id="BTC-USD",
    currency="BTC",
    interval="1h",
    instrument_type=InstrumentTypeConfig.CRYPTO,
    targetColumn="close",
    config=custom_config
)
```

### Adaptive Configuration
```python
from pipeline.timeSeriesProcessing.outlierDetection import build_config_from_properties

# Build adaptive config
params = {
    'instrument_type': InstrumentTypeConfig.CRYPTO,
    'interval': '1m',
    'data_length': 1000,
    'volatility': 0.35,
    'data_quality_score': 0.85,
    'outlier_ratio': 0.08
}

adaptive_config = build_config_from_properties(params)
# Config adapted for CRYPTO + HIGH_FREQ + LARGE data
```

---

> **Result**: Production-ready outlier detection with intelligent method selection, 95%+ context reuse, early stopping optimization, and comprehensive DataFrame enrichment for downstream processing.