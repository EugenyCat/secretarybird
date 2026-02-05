"""
Comprehensive Protocol for unified time series algorithm architecture.

Defines strict contract for all time series algorithms with mandatory configuration
patterns, workflow methods, strategy contracts, and error handling.
"""

import logging
from abc import abstractmethod
from typing import Any, ClassVar, Dict, Optional, Protocol, runtime_checkable

import pandas as pd

__version__ = "2.1.0"


@runtime_checkable
class BaseAlgorithm(Protocol):
    """
    Comprehensive Protocol for time series algorithms.

    IMPLEMENTATION STRATEGIES:
    - Consensus Weighted Voting: PeriodicityDetector (ensemble methods)
    - Enhanced Decision Tree: DecompositionAlgorithm (intelligent selection)
    - Simple Merge: TimeSeriesAnalyzer (unique results combination)

    COMPLIANCE: All algorithms MUST implement ALL mandatory methods with exact
    signatures and behavior contracts defined below.
    """

    # ========== MANDATORY CLASS ATTRIBUTES ==========

    AVAILABLE_METHODS: ClassVar[Dict[str, type]]
    """Registry of available method classes for dynamic instantiation."""

    # ========== MANDATORY INSTANCE ATTRIBUTES ==========

    config: Dict[str, Any]
    """Complete algorithm configuration with _active_methods and method configs."""

    enabled_methods: list[str]
    """Active methods extracted from config["_active_methods"]."""

    all_available_methods: list[str]
    """Complete list of available methods for the algorithm."""

    _methods: Dict[str, Any]
    """Lazy-loaded method instances cache. Initialize as {} in __init__."""

    _class_name: str
    """Cached class name for performance. Set as self.__class__.__name__."""

    # ========== MANDATORY INITIALIZATION ==========

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize algorithm following mandatory pattern.

        REQUIRED SEQUENCE:
        1. validate_required_locals(["config"], locals())
        2. Config None validation with ValueError
        3. Assign config, enabled_methods, all_available_methods
        4. Initialize _methods = {}, _class_name
        5. Algorithm-specific initialization
        6. Log initialization success

        Args:
            config: Algorithm configuration dictionary

        Raises:
            ValueError: If config is None or missing required fields
        """
        ...

    # ========== MANDATORY CORE WORKFLOW ==========

    @abstractmethod
    def process(
        self, data: pd.Series, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main algorithm execution with standardized workflow.

        MANDATORY PATTERN:
        1. try-catch wrapper with _handle_critical_error()
        2. Input validation via _validate_input()
        3. Method execution via _execute_methods()
        4. Result combination (strategy-dependent)
        5. Result finalization via _finalize_result()
        6. Success logging
        7. Return standardized response

        RESPONSE FORMAT:
        {
            'status': 'success' | 'error',
            'result': {...} | None,
            'message': str (errors only),
            'metadata': {
                'algorithm': str,
                'methods_used': list[str],
                'data_length': int,
                # Algorithm-specific metadata
            }
        }

        Args:
            data: Time series for processing
            context: Optional processing context

        Returns:
            Standardized response dictionary
        """
        ...

    @abstractmethod
    def _validate_input(
        self, data: pd.Series, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate input with standardized checks.

        MANDATORY CHECKS:
        1. Data None/empty validation
        2. Data type validation (pd.Series)
        3. Minimum length validation
        4. Algorithm-specific validation

        Args:
            data: Time series to validate
            context: Processing context (optional)

        Returns:
            {'status': 'success'} or error response dict
        """
        ...

    @abstractmethod
    def _execute_methods(
        self, data: pd.Series, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute algorithm methods according to strategy.

        STRATEGY PATTERNS:
        - Consensus: Execute ALL enabled methods for voting
        - Decision Tree: Execute baseline + selected method
        - Simple Merge: Execute ALL enabled methods for merge

        Returns:
            {
                'status': 'success' | 'error',
                'method_results': {method_name: result_dict},
                'errors': [error_dicts] (optional),
                'metadata': {
                    'strategy': str,
                    'methods_attempted': list[str],
                    'methods_succeeded': list[str]
                }
            }
        """
        ...

    @abstractmethod
    def _finalize_result(
        self,
        algorithm_result: Dict[str, Any],
        data: pd.Series,
        context: Optional[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Finalize result with comprehensive metadata.

        MANDATORY METADATA:
        - algorithm: self._class_name
        - data_length: len(data)
        - methods_used: list[str]
        - strategy: 'consensus' | 'decision_tree' | 'simple_merge'
        - Algorithm-specific fields

        Args:
            algorithm_result: Combined result from strategy
            data: Original input data
            context: Processing context
            **kwargs: Algorithm-specific parameters

        Returns:
            Finalized response with complete metadata
        """
        ...

    # ========== STRATEGY METHOD CONTRACTS ==========

    def _combine_results_consensus(
        self, method_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Consensus Weighted Voting for ensemble methods.

        REQUIRED FOR: PeriodicityDetector, TimeSeriesAnalyzer (simple merge variant)

        ALGORITHM:
        1. Extract candidates from method results
        2. Group similar candidates
        3. Apply weighted voting
        4. Add consensus bonus for agreement
        5. Cross-validate top candidates
        6. Select best with quality scoring

        Args:
            method_results: Results from all methods

        Returns:
            Combined result with consensus metrics

        Raises:
            NotImplementedError: If not using consensus strategy
        """
        raise NotImplementedError(
            f"{self._class_name} requires consensus strategy implementation"
        )

    def _combine_results_decision_tree(
        self,
        data: pd.Series,
        context: Optional[Dict[str, Any]],
        baseline_result: Dict[str, Any],
    ) -> str:
        """
        Enhanced Decision Tree for intelligent method selection.

        REQUIRED FOR: DecompositionAlgorithm

        PATTERN:
        1. Extract data characteristics from context
        2. Apply hierarchical decision rules
        3. Consider baseline quality
        4. Select optimal method
        5. Return method name

        Args:
            data: Time series for characteristics
            context: Context with analyzer/periodicity properties
            baseline_result: Baseline for quality reference

        Returns:
            Selected method name

        Raises:
            NotImplementedError: If not using decision tree strategy
        """
        raise NotImplementedError(
            f"{self._class_name} requires decision tree strategy implementation"
        )

    # ========== MANDATORY UTILITY METHODS ==========

    @abstractmethod
    def _get_method_instance(self, method_name: str):
        """
        Lazy-loading method retrieval with caching.

        PATTERN:
        1. Check cache in self._methods
        2. Validate method_name in AVAILABLE_METHODS
        3. Extract config from self.config[method_name]
        4. Instantiate and cache method
        5. Return cached instance

        Args:
            method_name: Method to instantiate

        Returns:
            Method instance

        Raises:
            ValueError: If method unknown or config missing
        """
        ...

    @abstractmethod
    def _handle_critical_error(self, error: Exception) -> Dict[str, Any]:
        """
        Standardized critical error handling.

        RESPONSE FORMAT:
        {
            'status': 'error',
            'message': 'Critical error in {algorithm}: {error}',
            'metadata': {
                'algorithm': str,
                'error_type': str,
                'error_stage': 'algorithm_execution'
            }
        }

        Args:
            error: Exception that caused failure

        Returns:
            Standardized error response
        """
        ...

    # ========== MANDATORY STRING REPRESENTATION ==========

    def __str__(self) -> str:
        """
        Standardized string representation.

        FORMAT: "{AlgorithmName}(methods={count})"

        Returns:
            String for consistent logging
        """
        return (
            f"{self._class_name}(methods={len(getattr(self, 'enabled_methods', []))})"
        )

    # ========== OPTIONAL ADVANCED FEATURES ==========

    def _validate_method_configuration(
        self, method_name: str, config: Dict[str, Any]
    ) -> bool:
        """
        Optional: Advanced method configuration validation.

        Categories: required parameters, types, ranges, consistency.

        Args:
            method_name: Method to validate
            config: Configuration to validate

        Returns:
            True if valid

        Raises:
            ValueError: If invalid with detailed message
        """
        return True

    def _calculate_performance_metrics(
        self, start_time: float, end_time: float, data_length: int
    ) -> Dict[str, float]:
        """
        Optional: Performance metrics for monitoring.

        Metrics: execution_time, throughput.

        Args:
            start_time: Start timestamp
            end_time: End timestamp
            data_length: Data points processed

        Returns:
            Performance metrics dict
        """
        execution_time = end_time - start_time
        throughput = data_length / execution_time if execution_time > 0 else 0.0

        return {
            "execution_time": execution_time,
            "throughput": throughput,
        }