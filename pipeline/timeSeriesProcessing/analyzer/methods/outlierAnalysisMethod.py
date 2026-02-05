"""
Outlier detection methods for time series analysis.

Implements three robust statistical methods:
- Robust Z-score (MAD-based, breakdown point 50%)
- IQR method (breakdown point 25%)
- MAD method (breakdown point 50%)

Mathematical references:
- Leys et al. (2013) "Detecting outliers: Do not use standard deviation"
- Rousseeuw & Croux (1993) "Alternatives to MAD"
- Tukey (1977) "Exploratory Data Analysis"

Version: 2.0.1 (Critical bugfix: counts dict type safety)
"""

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.helpers.utils import validate_required_locals
from pipeline.timeSeriesProcessing.analyzer.methods.baseAnalysisMethod import (
    BaseAnalysisMethod,
)

__version__ = "2.0.1"


class OutlierAnalysisMethod(BaseAnalysisMethod):
    """
    Robust outlier detection for time series.

    Mathematical approach:
    - Robust Z-score: MAD-based modified Z-score (Leys et al., 2013)
    - IQR method: Tukey's fence (1977)
    - MAD method: Median absolute deviation
    - Weighted consensus: Based on breakdown points (Rousseeuw & Croux, 1993)

    Breakdown points:
    - MAD/Robust Z-score: 50% (optimal robustness)
    - IQR: 25% (good robustness)
    - Classic Z-score: ~0% (non-robust, deprecated)
    """

    DEFAULT_CONFIG = {
        **BaseAnalysisMethod.DEFAULT_CONFIG,
        "use_robust_zscore": True,
        "min_robust_zscore_length": 10,
        "use_weighted_consensus": True,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize robust outlier detection method.

        Args:
            config: Configuration with parameters:
                - zscore_threshold: float
                - iqr_multiplier: float
                - mad_threshold: float
                - return_indices: bool
                - min_outliers_ratio: float
                - max_outliers_ratio: float
                - use_robust_zscore: bool (default True)
                - use_weighted_consensus: bool (default True)
        """
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged_config)

        validate_required_locals(
            [
                "zscore_threshold",
                "iqr_multiplier",
                "mad_threshold",
                "return_indices",
                "min_outliers_ratio",
                "max_outliers_ratio",
            ],
            self.config,
        )

    def __str__(self) -> str:
        """String representation for logging."""
        mode = "robust" if self.config["use_robust_zscore"] else "classic"
        return (
            f"OutlierAnalysisMethod(zscore={mode}, "
            f"threshold={self.config['zscore_threshold']}, "
            f"iqr_mult={self.config['iqr_multiplier']}, "
            f"mad_thresh={self.config['mad_threshold']})"
        )

    def process(
        self, data: pd.Series, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute robust outlier analysis on time series.

        Mathematical workflow:
        1. Robust Z-score detection (MAD-based)
        2. IQR method detection
        3. MAD method detection
        4. Weighted consensus (breakdown point based)
        5. Statistical metrics calculation

        Args:
            data: Time series for analysis
            context: Optional context with additional information

        Returns:
            Dict with standard format containing outlier counts, ratios, and statistics
        """
        try:
            validation = self.validate_input(data)
            if validation["status"] == "error":
                return validation

            context_params = self.extract_context_parameters(context)
            self.log_analysis_start(data, context_params)

            clean_data = self.prepare_clean_data(data, drop_na=True, min_length=3)

            result = {}
            outlier_indices = {}

            # 1. Robust Z-score method (MAD-based)
            zscore_result = self._robust_zscore_method(clean_data, self.config)
            result["zscore_outliers"] = zscore_result["count"]
            outlier_indices["zscore"] = zscore_result["indices"]
            result["zscore_method"] = zscore_result.get("method", "unknown")

            # 2. IQR method
            iqr_result = self._iqr_method(clean_data, self.config)
            result["iqr_outliers"] = iqr_result["count"]
            outlier_indices["iqr"] = iqr_result["indices"]

            # 3. MAD method
            mad_result = self._mad_method(clean_data, self.config)
            result["mad_outliers"] = mad_result["count"]
            outlier_indices["mad"] = mad_result["indices"]

            # 4. Weighted consensus (breakdown point based)
            # CRITICAL FIX: Create clean counts dict with only numeric values
            outlier_counts = {
                "zscore": result["zscore_outliers"],
                "iqr": result["iqr_outliers"],
                "mad": result["mad_outliers"],
            }
            combined_result = self._combine_outlier_results_weighted(
                data, outlier_counts, outlier_indices
            )

            # Check for errors in consensus calculation
            if "error" in combined_result:
                error_msg = f"Consensus calculation failed: {combined_result['error']}"
                logging.error(f"{self} - {error_msg}")
                return self._create_error_response(error_msg)

            result.update(combined_result)

            # 5. Outlier statistics
            outlier_stats = self._calculate_outlier_statistics(data, outlier_indices)
            result.update(outlier_stats)

            # Add indices if required
            if self.config["return_indices"]:
                result["outlier_indices"] = outlier_indices

            response = self.create_success_response(
                result,
                data,
                context_params,
                {
                    "detection_methods": ["robust_zscore", "iqr", "mad"],
                    "weighted_consensus": self.config["use_weighted_consensus"],
                },
            )

            self.log_analysis_complete(response)
            return response

        except Exception as e:
            return self.handle_error(e, "outlier analysis", {"data_length": len(data)})

    def _robust_zscore_method(
        self, data: pd.Series, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Robust Z-score outlier detection using MAD (Leys et al., 2013).

        Mathematical basis:
        - Classic Z-score: z = (x - mean) / std (NON-ROBUST)
        - Robust Z-score: z_mod = 0.6745 * |x - median| / MAD

        Constant 0.6745 = 1/(sqrt(2)*erf^(-1)(0.5)) for normality assumption.

        Args:
            data: Clean time series (no NaN)
            params: Config with zscore_threshold and use_robust_zscore

        Returns:
            Dict with count, indices, method, breakdown_point
        """
        use_robust = params.get("use_robust_zscore", True)
        min_length = params.get("min_robust_zscore_length", 10)

        if len(data) < min_length:
            logging.warning(
                f"{self} - Robust Z-score recommended for n≥{min_length}, "
                f"got n={len(data)}. Results may be unreliable."
            )

        if use_robust:
            if len(data) < 3:
                return {
                    "count": 0,
                    "indices": [],
                    "method": "robust_zscore_mad",
                    "breakdown_point": 0.50,
                }

            median = data.median()
            mad = np.median(np.abs(data - median))

            if mad < 1e-10:
                return {
                    "count": 0,
                    "indices": [],
                    "method": "robust_zscore_mad",
                    "warning": "zero_variation",
                }

            modified_z = 0.6745 * np.abs(data - median) / mad
            threshold = params["zscore_threshold"]
            outlier_mask = modified_z > threshold

            return {
                "count": int(outlier_mask.sum()),
                "indices": data[outlier_mask].index.tolist(),
                "threshold_used": threshold,
                "method": "robust_zscore_mad",
                "breakdown_point": 0.50,
            }
        else:
            from scipy.stats import zscore

            logging.warning(
                f"{self} - Using non-robust classic Z-score. "
                f"Not recommended! Set use_robust_zscore=True for production."
            )

            if len(data) < 30:
                logging.warning(
                    f"{self} - Classic Z-score requires n≥30, got n={len(data)}"
                )

            z_scores = np.abs(zscore(data))
            threshold = params["zscore_threshold"]
            outlier_mask = z_scores > threshold

            return {
                "count": int(outlier_mask.sum()),
                "indices": data[outlier_mask].index.tolist(),
                "threshold_used": threshold,
                "method": "classic_zscore_deprecated",
                "breakdown_point": 0.0,
                "warning": "non_robust_method",
            }

    def _iqr_method(self, data: pd.Series, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        IQR (Interquartile Range) outlier detection (Tukey, 1977).

        Mathematical formula:
        - IQR = Q3 - Q1
        - Lower bound = Q1 - k * IQR
        - Upper bound = Q3 + k * IQR
        - k = 1.5 (mild) or 3.0 (extreme)

        Breakdown point: 25%

        Args:
            data: Clean time series (no NaN)
            params: Config with iqr_multiplier

        Returns:
            Dict with count, indices, bounds, breakdown_point
        """
        if len(data) < 4:
            return {"count": 0, "indices": [], "breakdown_point": 0.25}

        q1, q3 = data.quantile([0.25, 0.75])
        iqr = q3 - q1

        multiplier = params["iqr_multiplier"]
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        outlier_mask = (data < lower_bound) | (data > upper_bound)
        outlier_indices = data[outlier_mask].index.tolist()

        return {
            "count": int(outlier_mask.sum()),
            "indices": outlier_indices,
            "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)},
            "breakdown_point": 0.25,
        }

    def _mad_method(self, data: pd.Series, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        MAD (Median Absolute Deviation) outlier detection.

        Mathematical formula:
        - MAD = median(|x - median(x)|)
        - Modified Z-score = 0.6745 * |x - median| / MAD

        Breakdown point: 50% (optimal robustness)

        Args:
            data: Clean time series (no NaN)
            params: Config with mad_threshold

        Returns:
            Dict with count, indices, mad, breakdown_point
        """
        if len(data) < 3:
            return {"count": 0, "indices": [], "breakdown_point": 0.50}

        median_val = data.median()
        mad = np.median(np.abs(data - median_val))

        if mad < 1e-10:
            return {
                "count": 0,
                "indices": [],
                "mad": 0.0,
                "warning": "zero_variation",
                "breakdown_point": 0.50,
            }

        modified_z = 0.6745 * np.abs(data - median_val) / mad
        threshold = params["mad_threshold"]
        outlier_mask = modified_z > threshold

        outlier_indices = data[outlier_mask].index.tolist()

        return {
            "count": int(outlier_mask.sum()),
            "indices": outlier_indices,
            "mad": float(mad),
            "threshold_used": threshold,
            "breakdown_point": 0.50,
        }

    def _combine_outlier_results_weighted(
        self, data: pd.Series, counts: Dict[str, int], indices: Dict[str, List]
    ) -> Dict[str, Any]:
        """
        Weighted consensus based on breakdown points (Rousseeuw & Croux, 1993).

        Mathematical approach:
        - Weight methods by robustness (breakdown points)
        - MAD: 50% → weight 0.50
        - IQR: 25% → weight 0.30
        - Robust Z-score: 50% → weight 0.20

        Consensus threshold: cumulative weight ≥ 0.5 (majority vote)

        Args:
            data: Original time series
            counts: Outlier counts per method (MUST be numeric only)
            indices: Outlier indices per method

        Returns:
            Dict with weighted consensus results
        """
        # Safety check: ensure counts contains only numeric values
        if not counts or not all(isinstance(v, (int, float)) for v in counts.values()):
            logging.error(
                f"{self} - Invalid counts dict: must contain only numeric values"
            )
            return {
                "outlier_ratio": 0.0,
                "consensus_outliers_count": 0,
                "detection_agreement": 0.0,
                "weighted_consensus": False,
                "error": "invalid_counts",
            }

        use_weighted = self.config.get("use_weighted_consensus", True)

        if use_weighted:
            weights = {
                "mad": 0.50,
                "iqr": 0.30,
                "zscore": 0.20,
            }

            index_scores = defaultdict(float)
            for method, method_indices in indices.items():
                weight = weights.get(method, 0)
                for idx in method_indices:
                    index_scores[idx] += weight

            consensus_indices = [
                idx for idx, score in index_scores.items() if score >= 0.5
            ]

            outlier_ratio = (
                len(consensus_indices) / len(data) if len(data) > 0 else 0.0
            )

            max_outliers = max(counts.values()) if counts and counts.values() else 0
            agreement = (
                len(consensus_indices) / max_outliers if max_outliers > 0 else 1.0
            )

            return {
                "outlier_ratio": outlier_ratio,
                "consensus_outliers_count": len(consensus_indices),
                "detection_agreement": agreement,
                "weighted_consensus": True,
                "method_weights": weights,
            }
        else:
            # Legacy: simple max-based consensus
            max_outliers = max(counts.values()) if counts and counts.values() else 0
            outlier_ratio = max_outliers / len(data) if len(data) > 0 else 0.0

            all_indices = []
            for method_indices in indices.values():
                all_indices.extend(method_indices)

            index_counts = Counter(all_indices)
            consensus_indices = [
                idx for idx, count in index_counts.items() if count >= 2
            ]

            agreement = (
                len(consensus_indices) / max_outliers if max_outliers > 0 else 1.0
            )

            return {
                "outlier_ratio": outlier_ratio,
                "max_outliers_count": max_outliers,
                "consensus_outliers_count": len(consensus_indices),
                "detection_agreement": agreement,
                "weighted_consensus": False,
            }

    def _calculate_outlier_statistics(
        self, data: pd.Series, outlier_indices: Dict[str, List]
    ) -> Dict[str, Any]:
        """
        Calculate additional outlier statistics.

        Metrics:
        - Variance ratio: outlier variance / normal variance
        - Average outlier distance: mean(|outlier - median_normal|)
        - Outlier clustering: std of gaps between outlier positions

        Args:
            data: Original time series
            outlier_indices: Outlier indices per method

        Returns:
            Dict with statistical metrics
        """
        all_outlier_idx = set()
        for indices in outlier_indices.values():
            all_outlier_idx.update(indices)

        if len(all_outlier_idx) == 0:
            return {
                "variance_ratio": 1.0,
                "avg_outlier_distance": 0.0,
                "outlier_clustering": 0.0,
            }

        outliers = data.loc[list(all_outlier_idx)]
        normal_data = data.drop(list(all_outlier_idx))

        if len(normal_data) == 0:
            return {
                "variance_ratio": np.inf,
                "avg_outlier_distance": 0.0,
                "outlier_clustering": 0.0,
            }

        # Variance ratio with numerical stability
        normal_var = normal_data.var()
        if normal_var < 1e-10:
            variance_ratio = np.inf if outliers.var() > 1e-10 else 1.0
        else:
            variance_ratio = float(outliers.var() / normal_var)

        # Average outlier distance
        median_val = normal_data.median()
        avg_outlier_distance = float(np.mean(np.abs(outliers - median_val)))

        # Outlier clustering
        outlier_clustering = 0.0
        if len(all_outlier_idx) > 1:
            try:
                sorted_indices = sorted(all_outlier_idx)
                position_indices = [data.index.get_loc(idx) for idx in sorted_indices]
                gaps = np.diff(position_indices)
                outlier_clustering = float(np.std(gaps)) if len(gaps) > 0 else 0.0
            except (KeyError, ValueError) as e:
                logging.warning(f"{self} - Could not calculate clustering: {e}")
                outlier_clustering = 0.0

        return {
            "variance_ratio": variance_ratio,
            "avg_outlier_distance": avg_outlier_distance,
            "outlier_clustering": outlier_clustering,
        }

    def process_with_dataframe_enrichment(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute outlier analysis with DataFrame enrichment.

        Adds boolean outlier columns:
        - is_zscore_outlier: Detected by robust Z-score
        - is_iqr_outlier: Detected by IQR method
        - is_mad_outlier: Detected by MAD method

        Args:
            dataframe: DataFrame with time series data
            target_column: Column name for analysis
            context: Optional context

        Returns:
            Tuple[enriched_dataframe, analysis_result]
        """
        try:
            validate_required_locals(["dataframe", "target_column"], locals())

            if target_column not in dataframe.columns:
                raise ValueError(f"Column '{target_column}' not found in DataFrame")

            data = dataframe[target_column]

            # Execute analysis with return_indices=True
            temp_config = self.config.copy()
            temp_config["return_indices"] = True
            temp_method = OutlierAnalysisMethod(temp_config)
            analysis_result = temp_method.process(data, context)

            if analysis_result["status"] != "success":
                return dataframe, analysis_result

            # Extract outlier indices
            outlier_indices = analysis_result["result"]["outlier_indices"]

            # Add boolean columns
            enriched_df = self._add_outlier_columns(dataframe, outlier_indices)

            logging.debug(
                f"{self} - Added outlier columns: "
                f"is_zscore_outlier, is_iqr_outlier, is_mad_outlier"
            )

            return enriched_df, analysis_result

        except Exception as e:
            error_msg = f"DataFrame enrichment failed: {str(e)}"
            logging.error(f"{self} - {error_msg}")
            return dataframe, {
                "status": "error",
                "message": error_msg,
                "metadata": {"error_type": type(e).__name__},
            }

    def _add_outlier_columns(
        self, dataframe: pd.DataFrame, outlier_indices: Dict[str, List[int]]
    ) -> pd.DataFrame:
        """Add boolean outlier columns to DataFrame."""
        enriched_df = dataframe.copy()

        # Initialize boolean columns (all False)
        enriched_df["is_zscore_outlier"] = False
        enriched_df["is_iqr_outlier"] = False
        enriched_df["is_mad_outlier"] = False

        # Set True for detected outliers
        if "zscore" in outlier_indices and outlier_indices["zscore"]:
            enriched_df.loc[outlier_indices["zscore"], "is_zscore_outlier"] = True

        if "iqr" in outlier_indices and outlier_indices["iqr"]:
            enriched_df.loc[outlier_indices["iqr"], "is_iqr_outlier"] = True

        if "mad" in outlier_indices and outlier_indices["mad"]:
            enriched_df.loc[outlier_indices["mad"], "is_mad_outlier"] = True

        return enriched_df