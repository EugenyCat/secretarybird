# BaseModule - Unified Architectural Foundation

> Universal base classes for standardizing development of time series processors and algorithms through Template Method pattern

## 🎯 Purpose

BaseModule contains the architectural foundation of the entire time series processing system. Implements Template Method pattern to drastically reduce code duplication and standardize workflow between all modules (analyzer, periodicity, decomposition).

Main role: provide unified interfaces and standardized patterns for developing new modules with minimal code duplication and maximum architectural consistency.

## 🔬 Architectural Principles

### Template Method Pattern
- **Standardized workflow**: common execution algorithm with delegation of specific logic
- **Abstract methods**: define extension points for module-specific functionality
- **Uniformity**: consistent approach to validation, logging, error handling
- **Extensibility**: new modules inherit ready-made architecture

### SOLID/DRY/KISS Principles
- **Single Responsibility**: each base class is responsible for its abstraction level
- **Open/Closed**: base classes closed for modification, open for extension
- **DRY**: elimination of 70-95% code duplication between modules
- **KISS**: minimalist interfaces with reasonable defaults

## 🏗️ Technical Implementation

**Architectural pattern**: 3-level hierarchy of base classes

**Main classes**:
- `BaseTimeSeriesMethod`: root base class for all methods (`baseMethod.py`)
- `BaseAlgorithm`: base class for algorithms (ensemble management) (`baseAlgorithm.py`)
- `BaseProcessor`: base class for processors (pipeline integration) (`baseProcessor.py`)
- `BaseConfigAdapter`: base class for configuration adapters (adaptive configuration) (`baseConfigAdapter.py`)

**Key patterns**:
- Template Method: standardized workflows with extension points
- Strategy Pattern: abstract methods for module-specific logic
- Lazy Loading: deferred initialization of heavy components
- Error Handling: unified error handling with fallback behavior

## 📁 File Structure

| File | Purpose | Key Components |
|------|---------|----------------|
| `baseMethod.py` | Root base class for methods | BaseTimeSeriesMethod, validate_input, error handling |
| `baseAlgorithm.py` | Base class for algorithms (Protocol) | BaseAlgorithm Protocol, comprehensive contracts |
| `baseProcessor.py` | Base class for processors | BaseProcessor, Template Method, DataFrame enrichment |
| `baseConfigAdapter.py` | Base class for configuration adapters | BaseConfigAdapter, Template Method, adaptive configuration|

> **Important**: Location of all base classes - `pipeline/timeSeriesProcessing/baseModule/`
> 
> - `BaseTimeSeriesMethod`: `baseMethod.py` (Implementation)
> - `BaseAlgorithm`: `baseAlgorithm.py` (Protocol)
> - `BaseProcessor`: `baseProcessor.py` (Implementation)

## 🔄 BaseTimeSeriesMethod - Level 1

### Purpose
Root base class for all time series processing methods. Eliminates duplication of common logic between analyzer, periodicity, decomposition modules.

### Key Capabilities
```python
class BaseTimeSeriesMethod(ABC):
    def validate_input(data, min_length) -> Dict[str, Any]
    def extract_context_parameters(context) -> Dict[str, Any]  
    def prepare_clean_data(data, drop_na=True) -> pd.Series
    def create_standard_metadata(data, context_params) -> Dict[str, Any]
    def create_success_response(result, data, context_params) -> Dict[str, Any]
    def handle_error(error, stage, metadata=None) -> Dict[str, Any]
    
    @abstractmethod
    def process(data, context) -> Dict[str, Any]
```

### Standardized Functions
- **Validation**: uniform input data checking
- **Context**: parameter extraction from context dict
- **Data Processing**: clean dataset preparation
- **Responses**: standardized result format
- **Logging**: unified for all methods

## 🔄 BaseAlgorithm - Level 2

### Purpose
Base class for time series algorithms. Manages method ensembles and implements Template Method for coordinating their execution. **Achieved 70% duplication reduction**.

### Template Method Workflow
```python
def process(data, context):
    1. Input data validation (_validate_input)
    2. Active method execution (_execute_methods) 
    3. Result combination (_combine_results) [ABSTRACT]
    4. Result post-processing (_post_process_results) [ABSTRACT]
```

### Abstract Methods (strategies)
```python
@abstractmethod
def _combine_results(method_results, data, context) -> Dict[str, Any]
    """Combination strategy: consensus voting, simple merge, decision tree"""

@abstractmethod  
def _post_process_results(combined_result, execution_result, data, context) -> Dict[str, Any]
    """Final post-processing and formatting"""

@abstractmethod
def _get_algorithm_name() -> str
    """Algorithm name for logging"""

@abstractmethod
def _validate_specific_input(data, context) -> Dict[str, Any]
    """Algorithm-specific validation"""
```

### Common Functionality
- **Lazy Loading**: `_get_method_instance()` for deferred method initialization
- **Ensemble Management**: active method execution with error handling
- **Critical Error Handling**: unified critical error processing

## 🔄 BaseProcessor - Level 3

### Purpose
High-level base class for time series processors. Pipeline integration with lifecycle management, repeated runs, DataFrame enrichment. **Achieved 95% duplication reduction**.

### Template Method Workflow
```python
def process(data, context):
    1. Input parameter validation
    2. Early return on repeated run → _handle_repeated_run()
    3. Module algorithm execution → _execute_algorithm() [ABSTRACT]
    4. DataFrame enrichment → _add_module_columns_to_dataframe()
    5. Property extraction → _extract_properties() [ABSTRACT]
    6. Context update with results
    7. Success logging → _log_success_summary() [ABSTRACT]
    8. Error handling → _handle_processor_error()
```

### Abstract Methods (strategies)
```python
@abstractmethod
def _execute_algorithm(series, context) -> Dict[str, Any]
    """Execute module algorithm"""

@abstractmethod
def _extract_properties(algorithm_result) -> Dict[str, Any]
    """Extract properties for DB storage"""

@abstractmethod
def _validate_properties(properties) -> bool
    """Validate existing properties from DB"""

@abstractmethod 
def _get_default_properties() -> Dict[str, Any]
    """Default properties for fallback"""

@abstractmethod
def _log_success_summary(properties) -> None
    """Module-specific success logging"""

@abstractmethod
def _initialize_algorithm()
    """Initialize module algorithm"""

@abstractmethod
def _restore_module_state() -> None
    """Restore state on repeated run"""

@abstractmethod
def _get_heuristic_fallback_values() -> Dict[str, Any]
    """Heuristic fallback values (SB8-68)"""
```

### Template Method Extensions
```python
def _restore_enrichment_columns(data, context) -> pd.DataFrame
    """OPTIONAL DataFrame enrichment restoration (Template Method Extension)"""
    # Default: no-op, override in modules with DataFrame enrichment
```

### Key Capabilities
- **10x Performance**: early return when properties exist
- **DataFrame Enrichment**: standardized column addition
- **Fallback Behaviors**: error/simple modes with graceful degradation
- **Repeated Runs**: properties reuse with enrichment
- **Heuristic Fallback**: module-specific heuristic values (SB8-68)

---

## 🔧 BaseConfigAdapter - Unified Configuration System

### Purpose
Base class for configuration adapters, implementing Template Method pattern to eliminate ~60% code duplication between configAnalyzer, configPeriodicity, configDecomposition.

### Architecture
```python
BaseConfigAdapter (Template Method - 9-step workflow)
    ↓
├── AnalyzerConfigAdapter
├── PeriodicityConfigAdapter
└── DecompositionConfigAdapter
```

### Unified Workflow
```python
def build_config_from_properties(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standardized 9-step configuration processing.
    """
    # 1. Input parameter validation (instrument_type, interval, data_length)
    # 2. Data classification (frequency, length, instrument + module-specific)
    # 3. Active method configuration initialization (from BASE + ACTIVE)
    # 4. Base parameter propagation (from BASE["base"])
    # 5. Adaptation rule application (frequency → length → instrument)
    # 6. Parameter constraint by data length (mathematical constraints)
    # 7. Parameter range validation (with auto-correction)
    # 8. Specific adjustment application (crypto, volatility, etc.)
    # 9. Configuration finalization (type conversion, rounding)
```

### Mandatory Class Attributes
```python
class YourConfigAdapter(BaseConfigAdapter):
    BASE: ClassVar[Dict[str, Dict[str, Any]]]    # Base method configurations
    ACTIVE: ClassVar[Dict[InstrumentTypeConfig, List[str]]]  # Active methods per instrument
    RULES: ClassVar[Dict[str, Dict[Any, List[Tuple[str, str, Any]]]]]  # Adaptation rules
    
    def _get_integer_parameter_map(self) -> Dict[str, List[str]]:
        """Mandatory implementation: integer parameter map"""
        return {"method1": ["param1", "param2"]}
```

### Hooks for Module Specificity
```python
# Pre-constraint setup (critical parameters BEFORE constraints)
def _pre_constraint_setup(self, config, data_length): pass

# Module-specific constraints
def _apply_module_specific_constraints(self, config, data_length): pass

# Module-specific validation
def _validate_module_specific_ranges(self, config): pass

# Crypto adjustments
def _apply_crypto_adjustments(self, config, volatility): pass

# Additional adaptations
def _apply_module_specific_adaptations(self, config, params, classifications): pass

# Finalization
def _finalize_module_specific(self, config, params): pass
```

### Key Capabilities
- **~60% Code Reduction**: elimination of duplication between config modules
- **Template Method**: standardized workflow with extension points
- **Adaptive Rules**: frequency/length/instrument adaptation rules
- **Mathematical Constraints**: automatic parameter constraint by data length
- **Auto-correction**: range validation with automatic correction
- **Crypto Adjustments**: specialized adjustments for cryptocurrencies

---

## 🔄 Module Integration

### Existing Module Inheritance
```python
# All processors inherit from BaseProcessor
class AnalysisProcessor(BaseProcessor):      # 451 → 80 lines (-82%)
class PeriodicityDetectorProcessor(BaseProcessor):  # 430 → 70 lines (-84%)  
class DecompositionProcessor(BaseProcessor): # 624 → 90 lines (-86%)

# All algorithms inherit from BaseAlgorithm  
class TimeSeriesAnalyzer(BaseAlgorithm):     # 205 lines (-46%)
class PeriodicityDetector(BaseAlgorithm):    # 426 lines (-29%)
class DecompositionAlgorithm(BaseAlgorithm): # 561 lines (-47%)

# All config adapters inherit from BaseConfigAdapter
class AnalyzerConfigAdapter(BaseConfigAdapter)
class PeriodicityConfigAdapter(BaseConfigAdapter)
class DecompositionConfigAdapter(BaseConfigAdapter)
```

### Creating New Module
```python
# 1. Create new processor
class YourProcessor(BaseProcessor):
    def __init__(self, ts_id, currency, interval, instrument_type, targetColumn, 
                 properties=None, config=None, fallbackBehavior="error"):
        super().__init__(ts_id=ts_id, currency=currency, interval=interval,
                        instrument_type=instrument_type, targetColumn=targetColumn,
                        properties=properties, config=config,
                        fallbackBehavior=fallbackBehavior, module_name="your_module")
    
    # Implement 8 abstract methods
    def _execute_algorithm(self, series, context): pass
    def _extract_properties(self, algorithm_result): pass
    # ... and others

# 2. Create algorithm  
class YourAlgorithm(BaseAlgorithm):
    # Implement 4 abstract strategies
    def _combine_results(self, method_results, data, context): pass
    def _post_process_results(self, combined_result, execution_result, data, context): pass
    # ... and others
```

## 🚀 Achieved Results

### Code Reduction Metrics
- **BaseProcessor**: 95% duplication elimination in processors
- **BaseAlgorithm**: 70% duplication elimination in algorithms  
- **Total Reduction**: ~1000+ lines of duplicated code eliminated

### Completed Tasks
- **SB8-65**: BaseProcessor creation and processor migration
- **SB8-66**: BaseAlgorithm creation and algorithm migration
- **SB8-67**: DecompositionAlgorithm TypeError fix
- **SB8-68**: Early return restoration and heuristic fallback

### Architectural Advantages
- **Standardization**: uniform interfaces for all modules
- **Maintainability**: base class changes inherited automatically
- **Extensibility**: new modules get ready-made architecture
- **Performance**: 10x boost through early return and Template Method optimizations
- **Quality**: unified standard for validation, logging, error handling

## 🔧 Guidelines

### For New Module Developers
1. **Always inherit** from appropriate base class
2. **Implement all abstract methods** according to Template Method contract
3. **Use validate_required_locals** at entry points
4. **Avoid .get()** - use [] for required keys
5. **Follow SOLID principles** when extending functionality

### For Base Class Modification
- **Changes must be backward compatible** 
- **New abstract methods require updating all child classes**
- **Template Method workflow must not be violated**
- **Thorough testing of all modules after changes**

---

> **Result**: unified architectural foundation ensures standardized development of time series modules with maximum code reuse and minimal duplication.