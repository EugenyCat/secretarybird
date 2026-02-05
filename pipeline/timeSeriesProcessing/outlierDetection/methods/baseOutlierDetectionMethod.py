"""
Base class for all outlier detection methods.

Provides:
- Boolean columns utilities for analyzer integration
- Consensus calculation across multiple detection methods
- Context-based parameter adaptation with numerical stability
- Decomposition components extraction
- Validation utilities for data quality
- Advanced methods: adaptive weighting, non-linear quality adjustment

Architectural Pattern:
- Inherits from BaseTimeSeriesMethod for DRY compliance
- Adds outlier-specific utilities (boolean columns, consensus, etc.)
- Follows SOLID/KISS/DRY principles
- Numerical stability enhancements (v1.1.0)

Mathematical Validation Status:
- Critical numerical stability issues fixed (v1.1.0)
- Autocorrelation adjustment validated (Brockwell & Davis 2016)
- Consensus mechanism validated (Kuncheva 2004)
"""

import logging
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from pipeline.helpers.utils import validate_required_locals
from pipeline.timeSeriesProcessing.baseModule.baseMethod import BaseTimeSeriesMethod

__version__ = "1.1.0"


class BaseOutlierDetectionMethod(BaseTimeSeriesMethod):
    """
    Base class for all outlier detection methods.

    Extends BaseTimeSeriesMethod with outlier-specific utilities:
    - Boolean columns utilities (extract, consensus, confidence)
    - Decomposition components extraction
    - Context-aware threshold adaptation with numerical stability
    - Validation utilities

    All outlier detection methods (Tier 1-4) inherit from this class.

    Version 1.1.0 Changes:
    - Fixed critical division by zero bugs
    - Added numerical stability safeguards
    - Implemented adaptive weighting (optional)
    - Implemented non-linear quality adjustment (optional)
    """

    DEFAULT_CONFIG = {
        **BaseTimeSeriesMethod.DEFAULT_CONFIG,
        "min_data_length": 30,
        "max_missing_ratio": 0.3,
        "consensus_threshold": 2,
        "confidence_threshold": 0.5,
    }
    NUMERICAL_EPSILON = 1e-10

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize outlier detection method.

        Args:
            config: Method configuration
        """
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged_config)

        validate_required_locals(
            ["min_data_length", "consensus_threshold"], self.config
        )

    @abstractmethod
    def detect(self, data: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect outliers in data.

        Args:
            data: DataFrame with price data and boolean columns
            context: Full context from analyzer/decomposition

        Returns:
            Dict with outliers, confidence, scores:
            {
                'outliers': pd.Series (boolean mask),
                'outlier_confidence': pd.Series (float [0, 1]),
                'outlier_count': int,
                'outlier_score': pd.Series (optional),
                'method_name': str,
                'metadata': {...}
            }
        """
        pass

    # ========================================
    # Section 1: Boolean Columns Utilities
    # ========================================

    def extract_boolean_outlier_masks(
        self, dataframe: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """
        Extract boolean outlier masks from DataFrame.

        Safely extracts is_zscore_outlier, is_iqr_outlier, is_mad_outlier
        columns created by analyzer module. Falls back to False if missing.

        Args:
            dataframe: DataFrame with boolean outlier columns

        Returns:
            Dict with boolean Series for each method:
            {
                'zscore': pd.Series (boolean),
                'iqr': pd.Series (boolean),
                'mad': pd.Series (boolean)
            }

        Example:
            >>> masks = method.extract_boolean_outlier_masks(df)
            >>> zscore_outliers = masks['zscore']
            >>> total_flagged = sum(mask.sum() for mask in masks.values())
        """
        masks = {}

        for method in ["zscore", "iqr", "mad"]:
            col_name = f"is_{method}_outlier"
            if col_name in dataframe.columns:
                masks[method] = dataframe[col_name].astype(bool)
            else:
                logging.warning(
                    f"{self} - Missing column: {col_name}, using False default"
                )
                masks[method] = pd.Series(False, index=dataframe.index)

        return masks

    # ========================================
    # Section 2: Confidence Scoring
    # ========================================

    def calculate_autocorrelation_adjusted_confidence(
        self,
        outlier_ratio: float,
        sample_size: int,
        lag1_autocorr: float,
        confidence_level: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Calculate autocorrelation-adjusted confidence interval for false positive rate.

        Uses Wilson score interval with effective sample size adjustment for
        temporal dependence (Brockwell & Davis 2016).

        Args:
            outlier_ratio: Observed outlier ratio p ∈ [0, 1]
            sample_size: Original sample size n
            lag1_autocorr: Lag-1 autocorrelation ρ
            confidence_level: CI level (default 0.95)

        Returns:
            Tuple of (lower_bound, upper_bound) for FPR

        Mathematical basis:
            - Effective sample size: n_eff = n * (1-ρ)/(1+ρ)  # Bartlett 1946
            - Wilson score interval with n_eff instead of n
            - Accounts for autocorrelation-induced variance inflation
        """
        # Validate inputs
        lag1_autocorr = self._validate_autocorrelation_bounds(lag1_autocorr)
        outlier_ratio = np.clip(outlier_ratio, 0.0, 1.0)

        # Calculate effective sample size (Bartlett 1946)
        n_eff = sample_size * (1 - lag1_autocorr) / (1 + lag1_autocorr)
        n_eff = max(n_eff, 10)  # Minimum for statistical validity

        # Wilson score interval
        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)

        # Calculate components
        center = outlier_ratio + z**2 / (2 * n_eff)
        margin = z * np.sqrt(
            outlier_ratio * (1 - outlier_ratio) / n_eff + z**2 / (4 * n_eff**2)
        )
        denominator = 1 + z**2 / n_eff

        # Bounds
        lower = (center - margin) / denominator
        upper = (center + margin) / denominator

        return (float(np.clip(lower, 0.0, 1.0)), float(np.clip(upper, 0.0, 1.0)))

    def calculate_consensus_outliers(
        self, boolean_masks: Dict[str, pd.Series], threshold: int = 2
    ) -> pd.Series:
        """
        Calculate consensus outliers (N+ methods agree).

        Implements majority voting: point is outlier if threshold+ methods
        agree. Default threshold=2 means 2 out of 3 methods must agree.

        Args:
            boolean_masks: Dict of boolean Series from different methods
            threshold: Minimum number of methods that must agree (default: 2)

        Returns:
            Boolean Series of consensus outliers

        Example:
            >>> masks = {'zscore': series1, 'iqr': series2, 'mad': series3}
            >>> consensus = method.calculate_consensus_outliers(
            ...     masks, threshold=2
            ... )
            >>> print(f"Consensus outliers: {consensus.sum()}")
        """
        if not boolean_masks:
            logging.warning(f"{self} - Empty boolean_masks, returning all False")
            return pd.Series(False, index=range(0), dtype=bool)

        # Get first mask for index reference
        first_mask = next(iter(boolean_masks.values()))

        # Initialize consensus count
        consensus_count = pd.Series(0, index=first_mask.index, dtype=int)

        # Sum boolean masks (True=1, False=0)
        for mask in boolean_masks.values():
            consensus_count += mask.astype(int)

        # Threshold: N+ methods agree
        consensus = consensus_count >= threshold

        logging.debug(
            f"{self} - Consensus outliers: {int(consensus.sum())} "
            f"(threshold={threshold}/{len(boolean_masks)})"
        )

        return consensus

    def calculate_confidence_scores(
        self,
        boolean_masks: Dict[str, pd.Series],
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """
        Calculate weighted confidence scores from boolean masks.

        Confidence score [0, 1] indicates how strongly multiple methods agree.
        Higher score = more methods flagged the point as outlier.

        NUMERICAL STABILITY (v1.1.0):
        - Protected against division by zero
        - Fallback to equal weights if sum(weights) <= 0
        - Validated against Mathematical Validation Report

        Args:
            boolean_masks: Dict of boolean Series from different methods
            weights: Optional dict of weights for each method.
                     If None, equal weights used. Defaults to None.

        Returns:
            Float Series [0, 1] of confidence scores

        Example:
            >>> masks = {'zscore': series1, 'iqr': series2, 'mad': series3}
            >>> weights = {'zscore': 1.5, 'iqr': 1.0, 'mad': 1.0}
            >>> confidence = method.calculate_confidence_scores(
            ...     masks, weights
            ... )
            >>> high_confidence = confidence[confidence > 0.7]
        """
        if not boolean_masks:
            logging.warning(f"{self} - Empty boolean_masks, returning zeros")
            return pd.Series(0.0, index=range(0), dtype=float)

        # Get first mask for index reference
        first_mask = next(iter(boolean_masks.values()))

        # Build weights list
        all_weights = []
        for method in boolean_masks.keys():
            weight = weights[method] if weights else 1.0
            all_weights.append(weight)

        # CRITICAL FIX v1.1.0: Check sum(weights) > 0 before division
        total_weights_sum = sum(all_weights)
        if total_weights_sum <= 0:
            logging.error(
                f"{self} - Invalid weights: sum={total_weights_sum}. "
                "Using equal weights fallback."
            )
            # Fallback to equal weights
            all_weights = [1.0] * len(boolean_masks)
            total_weights_sum = len(boolean_masks)

        # Calculate weighted sum
        total_weight = pd.Series(0.0, index=first_mask.index, dtype=float)

        for (method, mask), weight in zip(boolean_masks.items(), all_weights):
            total_weight += mask.astype(float) * weight

        # Now safe to divide (guaranteed total_weights_sum > 0)
        confidence = total_weight / total_weights_sum

        logging.debug(f"{self} - Mean confidence: {float(confidence.mean()):.3f}")

        return confidence

    def calculate_adaptive_confidence_scores(
        self,
        boolean_masks: Dict[str, pd.Series],
        method_reliability: Optional[Dict[str, float]] = None,
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """
        Calculate confidence scores with adaptive weights based on reliability.

        OPTIONAL METHOD (v1.1.0): Use when historical method reliability known.

        Adaptive weighting strategy:
        - Higher weight for methods with historically better precision/recall
        - Dynamic adjustment based on data characteristics
        - Validated against Kuncheva (2004) ensemble methods theory

        Args:
            boolean_masks: Dict of boolean Series from different methods
            method_reliability: Optional dict of historical reliability scores
                                [0, 1]. Higher = more reliable method.

        Returns:
            Tuple of (confidence_scores, used_weights)

        Example:
            >>> masks = {'zscore': s1, 'iqr': s2, 'mad': s3}
            >>> reliability = {'zscore': 0.85, 'iqr': 0.75, 'mad': 0.80}
            >>> confidence, weights = (
            ...     method.calculate_adaptive_confidence_scores(
            ...         masks, reliability
            ...     )
            ... )
            >>> print(f"Used weights: {weights}")
        """
        if not boolean_masks:
            return pd.Series(0.0, index=range(0)), {}

        # Calculate adaptive weights
        weights = {}
        for method in boolean_masks.keys():
            if method_reliability and method in method_reliability:
                # Validate reliability in [0, 1]
                reliability = method_reliability[method]
                reliability = np.clip(reliability, 0.0, 1.0)
                weights[method] = reliability
            else:
                # Fallback to equal weights
                weights[method] = 1.0

        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            # Equal weights fallback
            n = len(boolean_masks)
            weights = {k: 1.0 / n for k in boolean_masks.keys()}

        # Calculate weighted confidence (now safe - weights sum to 1)
        confidence = self.calculate_confidence_scores(boolean_masks, weights)

        return confidence, weights

    # ========================================
    # Section 3: Numerical Validation Helpers
    # ========================================

    def _validate_autocorrelation_bounds(self, lag1_autocorr: float) -> float:
        """
        Validate and clip autocorrelation to valid range.

        Statistical theory requires ρ ∈ (-1, 1). This method ensures
        numerical stability by clipping to [-0.95, 0.95] with margin.

        Args:
            lag1_autocorr: Lag-1 autocorrelation value

        Returns:
            Clipped autocorrelation value

        Mathematical Basis:
            - Brockwell & Davis (2016) - autocorrelation properties
            - Margin at 0.95 prevents numerical issues at boundary
        """
        if abs(lag1_autocorr) >= 1.0:
            logging.warning(
                f"{self} - Invalid lag1_autocorr={lag1_autocorr:.3f} "
                "(must be in (-1, 1)). Clipping to ±0.95"
            )
            return np.clip(lag1_autocorr, -0.95, 0.95)

        # Also clip if very close to boundary
        if abs(lag1_autocorr) > 0.95:
            logging.info(
                f"{self} - High lag1_autocorr={lag1_autocorr:.3f}, "
                "clipping to ±0.95 for numerical stability"
            )
            return np.clip(lag1_autocorr, -0.95, 0.95)

        return lag1_autocorr

    def _calculate_quality_factor(self, quality_score: float) -> float:
        """
        Calculate quality adjustment factor with non-linear scaling.

        OPTIONAL METHOD (v1.1.0): Non-linear adjustment provides stronger
        penalty for low quality data than linear approach.

        Mathematical basis:
        - Linear adjustment insufficient for very poor quality
        - Non-linear (sqrt inverse) provides stronger penalty
        - Preserves threshold stability for high-quality data

        Formula: quality_factor = 1.0 / sqrt(quality_score)
        - quality=1.0 → factor=1.0 (no adjustment)
        - quality=0.5 → factor=1.41 (41% increase)
        - quality=0.25 → factor=2.0 (100% increase)

        Args:
            quality_score: Data quality score [0, 1]

        Returns:
            Quality adjustment factor [1.0, 3.0]

        Example:
            >>> factor_high = method._calculate_quality_factor(0.9)
            >>> factor_low = method._calculate_quality_factor(0.3)
            >>> print(f"High quality factor: {factor_high:.2f}")  # ~1.05
            >>> print(f"Low quality factor: {factor_low:.2f}")    # ~1.83
        """
        # CRITICAL FIX v1.1.0: Validate bounds and avoid division by zero
        quality_score = np.clip(quality_score, self.NUMERICAL_EPSILON, 1.0)

        # Non-linear adjustment: stronger penalty for low quality
        quality_factor = 1.0 / np.sqrt(quality_score)

        # Cap maximum adjustment at 3x for stability
        quality_factor = min(quality_factor, 3.0)

        return quality_factor

    # ========================================
    # Section 4: Context Adaptation
    # ========================================

    def adapt_threshold_to_context(
        self, base_threshold: float, context: Dict[str, Any]
    ) -> float:
        """
        Adapt threshold based on data context.

        Adjusts threshold based on:
        - Data quality score (lower quality → higher threshold)
        - Autocorrelation (higher autocorr → adjusted threshold)
        - Volatility (implicit via quality score)

        NUMERICAL STABILITY (v1.1.0):
        - Protected against division by zero
        - Validated autocorrelation bounds
        - Safe sqrt operations
        - Validated against Mathematical Validation Report

        Mathematical basis:
        - Brockwell & Davis (2016) for autocorrelation adjustment
        - Variance inflation factor: sqrt(1 + 2*ρ/(1-ρ)) for |ρ| > 0.5

        Args:
            base_threshold: Base threshold value
            context: Context with analyzer properties

        Returns:
            Adjusted threshold value

        Example:
            >>> base_z_threshold = 3.0
            >>> adjusted = method.adapt_threshold_to_context(
            ...     base_z_threshold, context
            ... )
            >>> print(f"Adjusted threshold: {adjusted:.2f}")
        """
        try:
            analyzer = context["currentProperties"]["analyzer"]
        except KeyError:
            logging.warning(
                f"{self} - Missing analyzer in context, using base threshold"
            )
            return base_threshold

        # Quality adjustment: lower quality → higher threshold
        quality_score = analyzer["data_quality_score"]

        # CRITICAL FIX v1.1.0: Validate quality_score bounds
        quality_score = np.clip(quality_score, 0.0, 1.0)

        # Use linear adjustment for simplicity (can use non-linear via helper)
        quality_factor = 1.0 + (1.0 - quality_score) * 0.2  # Up to 20%

        # Autocorrelation adjustment (Brockwell & Davis, 2016)
        lag1_autocorr = analyzer["lag1_autocorrelation"]

        # CRITICAL FIX v1.1.0: Validate autocorrelation bounds BEFORE formula
        lag1_autocorr = self._validate_autocorrelation_bounds(lag1_autocorr)

        # Only apply adjustment for significant autocorrelation
        if abs(lag1_autocorr) > 0.5:
            # CRITICAL FIX v1.1.0: Additional safety checks
            numerator = 1 + lag1_autocorr
            denominator = 1 - lag1_autocorr

            # Check denominator (should be > 0 after clipping, but verify)
            if abs(denominator) < self.NUMERICAL_EPSILON:
                logging.warning(
                    f"{self} - Near-zero denominator in autocorr formula. "
                    "Using max factor 2.0"
                )
                autocorr_factor = 2.0
            else:
                formula_input = numerator / denominator

                # CRITICAL FIX v1.1.0: Validate sqrt input
                if formula_input < 0:
                    logging.error(
                        f"{self} - Negative sqrt input: {formula_input:.3f}. "
                        "This indicates invalid autocorrelation. "
                        "Using factor 1.0"
                    )
                    autocorr_factor = 1.0
                else:
                    autocorr_factor = np.sqrt(formula_input)
                    autocorr_factor = min(autocorr_factor, 2.0)  # Cap at 2x
        else:
            autocorr_factor = 1.0

        # Combined adjustment
        adjusted_threshold = base_threshold * quality_factor * autocorr_factor

        logging.debug(
            f"{self} - Threshold adaptation: "
            f"base={base_threshold:.2f}, "
            f"quality_factor={quality_factor:.2f}, "
            f"autocorr_factor={autocorr_factor:.2f}, "
            f"adjusted={adjusted_threshold:.2f}"
        )

        return adjusted_threshold

    # ========================================
    # Section 5: Component-Aware Processing
    # ========================================

    def extract_decomposition_components(
        self, dataframe: pd.DataFrame, target_column: str = "close"
    ) -> Dict[str, Optional[pd.Series]]:
        """
        Extract decomposition components from DataFrame.

        Safely extracts trend, seasonal, residual components created by
        decomposition module. Returns None for missing components.

        Args:
            dataframe: DataFrame with decomposition components
            target_column: Target column name (e.g., 'close', 'volume')

        Returns:
            Dict with component Series (or None if missing):
            {
                'trend': pd.Series or None,
                'seasonal': pd.Series or None,
                'residual': pd.Series or None
            }

        Example:
            >>> components = method.extract_decomposition_components(
            ...     df, 'close'
            ... )
            >>> if components['trend'] is not None:
            ...     print(f"Trend length: {len(components['trend'])}")
        """
        components = {}

        component_names = ["trend", "seasonal", "residual"]
        for comp_name in component_names:
            col_name = f"{comp_name}"

            if col_name in dataframe.columns:
                components[comp_name] = dataframe[col_name]
            else:
                logging.debug(f"{self} - Missing component: {col_name}, using None")
                components[comp_name] = None

        return components

    # ========================================
    # Section 6: Advanced Methods (Optional)
    # ========================================

    def validate_method_independence(
        self,
        boolean_masks: Dict[str, pd.Series],
        max_correlation: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Validate independence assumption for ensemble methods.

        OPTIONAL METHOD (v1.1.0): Use for diagnostic purposes.

        High correlation between methods reduces ensemble diversity
        and may bias consensus toward redundant information.

        Args:
            boolean_masks: Dict of boolean Series from different methods
            max_correlation: Maximum acceptable correlation threshold

        Returns:
            Dict with correlation matrix and independence validation:
            {
                'status': 'independent' or 'correlated',
                'correlation_matrix': np.ndarray,
                'methods': List[str],
                'high_correlations': List[Dict],
                'max_correlation': float
            }

        Example:
            >>> masks = {'zscore': s1, 'iqr': s2, 'mad': s3}
            >>> result = method.validate_method_independence(masks)
            >>> if result['status'] == 'correlated':
            ...     print(f"High correlations: {result['high_correlations']}")
        """
        if len(boolean_masks) < 2:
            return {
                "status": "insufficient_methods",
                "correlation_matrix": None,
            }

        # Calculate pairwise correlations
        methods = list(boolean_masks.keys())
        n = len(methods)
        corr_matrix = np.zeros((n, n))

        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    mask1 = boolean_masks[method1].astype(float)
                    mask2 = boolean_masks[method2].astype(float)
                    # Check for constant masks first
                    if (
                        mask1.std() < self.NUMERICAL_EPSILON
                        or mask2.std() < self.NUMERICAL_EPSILON
                    ):
                        corr_matrix[i, j] = 1.0 if i == j else 0.0
                    else:
                        # Use Spearman rank correlation (robust)
                        try:
                            corr_result = stats.spearmanr(mask1, mask2)
                            corr = corr_result.correlation
                            # Handle NaN
                            if np.isnan(corr) or np.isinf(corr):
                                corr = 0.0
                        except:
                            corr = 0.0

                        corr_matrix[i, j] = corr

        # Find high correlations
        high_corr_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr_matrix[i, j]) > max_correlation:
                    high_corr_pairs.append(
                        {
                            "method1": methods[i],
                            "method2": methods[j],
                            "correlation": corr_matrix[i, j],
                        }
                    )

        is_independent = len(high_corr_pairs) == 0

        return {
            "status": "independent" if is_independent else "correlated",
            "correlation_matrix": corr_matrix,
            "methods": methods,
            "high_correlations": high_corr_pairs,
            "max_correlation": max(
                abs(corr_matrix[i, j]) for i in range(n) for j in range(i + 1, n)
            ),
        }
