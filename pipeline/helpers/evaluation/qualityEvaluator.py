"""
Universal quality evaluation system for time series.

Supports decomposition, forecasting, and future ML/NN tasks.
Eliminates MSE/MAE/AIC metric duplication across various modules.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox

from pipeline.helpers.configs import QualityMetricConfig
from pipeline.helpers.utils import validate_required_locals

__version__ = "2.0.0"


class QualityEvaluator:
    """
    Universal quality evaluation system for time series.

    Supports:
    - Decomposition
    - Prediction
    - Custom configurations

    Eliminates MSE/MAE/AIC duplication in the project.
    """

    # Weight configurations for various task types
    WEIGHT_CONFIGS = {
        "decomposition": {
            QualityMetricConfig.MSE: -0.3,  # Lower = better
            QualityMetricConfig.MAE: -0.3,  # Lower = better
            QualityMetricConfig.AIC: -0.1,  # Lower = better
            QualityMetricConfig.BIC: -0.1,  # Lower = better
            QualityMetricConfig.RESIDUAL_AUTOCORR: 0.2,  # Higher = better
            QualityMetricConfig.SEASONAL_STRENGTH: 0.1,
            QualityMetricConfig.TREND_STRENGTH: 0.1,
            QualityMetricConfig.ROBUSTNESS: 0.2,
        },
        "prediction": {
            QualityMetricConfig.MSE: -0.4,  # Forecast accuracy priority
            QualityMetricConfig.MAE: -0.4,  # Forecast accuracy priority
            QualityMetricConfig.AIC: -0.1,
            QualityMetricConfig.BIC: -0.1,
        },
        "custom": {},  # Set by user
    }

    def __init__(
        self,
        evaluation_type: str = "decomposition",
        custom_weights: Optional[Dict[QualityMetricConfig, float]] = None,
    ):
        """
        Initialize universal quality evaluator.

        Args:
            evaluation_type: Evaluation type ('decomposition', 'prediction', 'custom')
            custom_weights: Custom weights for evaluation_type='custom'
        """
        validate_required_locals(["evaluation_type"], locals())

        self.evaluation_type = evaluation_type

        if evaluation_type == "custom":
            if custom_weights is None:
                raise ValueError("custom_weights required for evaluation_type='custom'")
            self.metric_weights = custom_weights
        else:
            if evaluation_type not in self.WEIGHT_CONFIGS:
                raise ValueError(f"Unsupported evaluation_type: {evaluation_type}")
            self.metric_weights = self.WEIGHT_CONFIGS[evaluation_type]

    def __str__(self) -> str:
        """Standard string representation for logging."""
        return f"QualityEvaluator(type={self.evaluation_type}, metrics={len(self.metric_weights)})"

    # ============================================================================
    # DECOMPOSITION EVALUATION (Main functionality for decomposition)
    # ============================================================================

    def evaluate_decomposition(
        self,
        original: pd.Series,
        trend: pd.Series,
        seasonal: pd.Series,
        residual: pd.Series,
        metrics: Optional[List[QualityMetricConfig]] = None,
    ) -> Dict[str, float]:
        """
        Comprehensive decomposition quality evaluation.

        Args:
            original: Original time series
            trend: Trend component
            seasonal: Seasonal component
            residual: Residual component
            metrics: List of metrics to compute

        Returns:
            Dictionary with quality scores
        """
        validate_required_locals(
            ["original", "trend", "seasonal", "residual"], locals()
        )

        if metrics is None:
            metrics = [
                QualityMetricConfig.MSE,
                QualityMetricConfig.SEASONAL_STRENGTH,
                QualityMetricConfig.RESIDUAL_AUTOCORR,
            ]

        scores = {}

        # Reconstructed series
        reconstructed = trend + seasonal + residual

        # Compute metrics
        for metric in metrics:
            try:
                score = self._calculate_decomposition_metric(
                    metric, original, reconstructed, trend, seasonal, residual
                )
                scores[metric.value] = score
            except Exception as e:
                logging.warning(f"Error computing {metric.value}: {e}")
                scores[metric.value] = 0.0

        # Composite score
        scores["composite_score"] = self._calculate_composite_score(scores)

        return scores

    def evaluate_decomposition_single_metric(
        self,
        metric: QualityMetricConfig,
        original: pd.Series,
        trend: pd.Series,
        seasonal: pd.Series,
        residual: pd.Series,
    ) -> float:
        """
        Compute a single decomposition quality metric.

        Args:
            metric: Metric to compute
            original: Original time series
            trend: Trend component
            seasonal: Seasonal component
            residual: Residual component

        Returns:
            Metric value
        """
        validate_required_locals(
            ["metric", "original", "trend", "seasonal", "residual"], locals()
        )

        reconstructed = trend + seasonal + residual
        return self._calculate_decomposition_metric(
            metric, original, reconstructed, trend, seasonal, residual
        )

    # ============================================================================
    # PREDICTION EVALUATION (For future ML/NN tasks)
    # ============================================================================

    def evaluate_prediction(
        self,
        actual: Union[pd.Series, np.ndarray],
        predicted: Union[pd.Series, np.ndarray],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Prediction quality evaluation for ML/NN models.

        Args:
            actual: Actual values
            predicted: Predicted values
            metrics: List of metrics ('mse', 'mae', 'mape', 'r2')

        Returns:
            Dictionary with forecast quality metrics
        """
        validate_required_locals(["actual", "predicted"], locals())

        if metrics is None:
            metrics = ["mse", "mae", "mape", "r2"]

        # Convert to numpy for universality
        if isinstance(actual, pd.Series):
            actual = actual.values
        if isinstance(predicted, pd.Series):
            predicted = predicted.values

        scores = {}

        for metric in metrics:
            try:
                if metric == "mse":
                    scores["mse"] = self.calculate_mse(actual, predicted)
                elif metric == "mae":
                    scores["mae"] = self.calculate_mae(actual, predicted)
                elif metric == "mape":
                    scores["mape"] = self.calculate_mape(actual, predicted)
                elif metric == "r2":
                    scores["r2"] = self.calculate_r2(actual, predicted)
                else:
                    logging.warning(f"Unknown prediction metric: {metric}")
                    scores[metric] = 0.0

            except Exception as e:
                logging.warning(f"Error calculating {metric}: {e}")
                scores[metric] = 0.0

        return scores

    def evaluate_prediction_single_metric(
        self,
        metric: str,
        actual: Union[pd.Series, np.ndarray],
        predicted: Union[pd.Series, np.ndarray],
    ) -> float:
        """
        Compute a single forecast quality metric.

        Args:
            metric: Metric name ('mse', 'mae', 'mape', 'r2')
            actual: Actual values
            predicted: Predicted values

        Returns:
            Metric value
        """
        result = self.evaluate_prediction(actual, predicted, [metric])
        return result.get(metric, 0.0)

    # ============================================================================
    # UNIVERSAL METRICS (Reusable computations)
    # ============================================================================

    def calculate_mse(
        self,
        actual: Union[pd.Series, np.ndarray],
        predicted: Union[pd.Series, np.ndarray],
    ) -> float:
        """Mean Squared Error (MSE)."""
        if isinstance(actual, pd.Series):
            actual = actual.values
        if isinstance(predicted, pd.Series):
            predicted = predicted.values

        return float(np.mean((actual - predicted) ** 2))

    def calculate_mae(
        self,
        actual: Union[pd.Series, np.ndarray],
        predicted: Union[pd.Series, np.ndarray],
    ) -> float:
        """Mean Absolute Error (MAE)."""
        if isinstance(actual, pd.Series):
            actual = actual.values
        if isinstance(predicted, pd.Series):
            predicted = predicted.values

        return float(np.mean(np.abs(actual - predicted)))

    def calculate_mape(
        self,
        actual: Union[pd.Series, np.ndarray],
        predicted: Union[pd.Series, np.ndarray],
    ) -> float:
        """Mean Absolute Percentage Error (MAPE)."""
        if isinstance(actual, pd.Series):
            actual = actual.values
        if isinstance(predicted, pd.Series):
            predicted = predicted.values

        # Avoid division by zero
        mask = actual != 0
        if not np.any(mask):
            return 100.0

        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        return float(mape if not np.isnan(mape) and not np.isinf(mape) else 100.0)

    def calculate_r2(
        self,
        actual: Union[pd.Series, np.ndarray],
        predicted: Union[pd.Series, np.ndarray],
    ) -> float:
        """Coefficient of Determination (R²)."""
        if isinstance(actual, pd.Series):
            actual = actual.values
        if isinstance(predicted, pd.Series):
            predicted = predicted.values

        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)

        return float(1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0)

    def calculate_aic(
        self,
        actual: Union[pd.Series, np.ndarray],
        predicted: Union[pd.Series, np.ndarray],
        n_parameters: int = 3,
    ) -> float:
        """Akaike Information Criterion (AIC)."""
        if isinstance(actual, pd.Series):
            actual = actual.values
        if isinstance(predicted, pd.Series):
            predicted = predicted.values

        residuals = actual - predicted
        n = len(actual)

        sse = np.sum(residuals**2)
        if sse <= 0:
            return float("inf")

        # AIC = n * ln(SSE/n) + 2k
        aic = n * np.log(sse / n) + 2 * n_parameters
        return float(aic)

    def calculate_bic(
        self,
        actual: Union[pd.Series, np.ndarray],
        predicted: Union[pd.Series, np.ndarray],
        n_parameters: int = 3,
    ) -> float:
        """Bayesian Information Criterion (BIC)."""
        if isinstance(actual, pd.Series):
            actual = actual.values
        if isinstance(predicted, pd.Series):
            predicted = predicted.values

        residuals = actual - predicted
        n = len(actual)

        sse = np.sum(residuals**2)
        if sse <= 0:
            return float("inf")

        # BIC = n * ln(SSE/n) + k * ln(n)
        bic = n * np.log(sse / n) + n_parameters * np.log(n)
        return float(bic)

    # ============================================================================
    # PRIVATE METHODS (Internal logic)
    # ============================================================================

    def _calculate_decomposition_metric(
        self,
        metric: QualityMetricConfig,
        original: pd.Series,
        reconstructed: pd.Series,
        trend: pd.Series,
        seasonal: pd.Series,
        residual: pd.Series,
    ) -> float:
        """Compute a specific decomposition metric."""
        if metric == QualityMetricConfig.MSE:
            return self.calculate_mse(original, reconstructed)
        elif metric == QualityMetricConfig.MAE:
            return self.calculate_mae(original, reconstructed)
        elif metric == QualityMetricConfig.AIC:
            return self.calculate_aic(original, reconstructed)
        elif metric == QualityMetricConfig.BIC:
            return self.calculate_bic(original, reconstructed)
        elif metric == QualityMetricConfig.RESIDUAL_AUTOCORR:
            return self._test_residual_autocorrelation(residual)
        elif metric == QualityMetricConfig.SEASONAL_STRENGTH:
            return self._calculate_seasonal_strength(original, seasonal, residual)
        elif metric == QualityMetricConfig.TREND_STRENGTH:
            return self._calculate_trend_strength(original, trend, residual, seasonal)
        elif metric == QualityMetricConfig.ROBUSTNESS:
            return self._robustness_test(original, residual)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _test_residual_autocorrelation(self, residual: pd.Series) -> float:
        """
        Test residuals for autocorrelation.

        Returns p-value of Ljung-Box test.
        High value (close to 1) means no autocorrelation (good).
        """
        try:
            clean_residual = residual.dropna()
            if len(clean_residual) < 10:
                return 0.5

            # Number of lags for the test
            lags = min(10, len(clean_residual) // 4)

            # Ljung-Box test
            lb_result = acorr_ljungbox(clean_residual, lags=lags, return_df=True)

            # Return minimum p-value
            min_pvalue = lb_result["lb_pvalue"].min()
            return float(min_pvalue)

        except Exception as e:
            logging.warning(f"Error in autocorrelation test: {e}")
            return 0.5

    def _calculate_seasonal_strength(
        self, original: pd.Series, seasonal: pd.Series, residual: pd.Series
    ) -> float:
        """
        Calculate seasonal component strength using Hyndman formula.

        Seasonal strength = 1 - Var(residual) / Var(seasonal + residual)
        """
        seasonal_plus_residual = seasonal + residual
        var_seasonal_residual = np.var(seasonal_plus_residual)

        if var_seasonal_residual == 0:
            return 0.0

        var_residual = np.var(residual)
        strength = max(0.0, 1.0 - var_residual / var_seasonal_residual)

        return float(strength)

    def _calculate_trend_strength(
        self,
        original: pd.Series,
        trend: pd.Series,
        residual: pd.Series,
        seasonal: pd.Series,
    ) -> float:
        """
        Calculate trend component strength using Hyndman formula.

        Trend strength = 1 - Var(residual) / Var(detrended)
        where detrended = original - seasonal
        """
        detrended = original - seasonal
        var_detrended = np.var(detrended)

        if var_detrended == 0:
            return 0.0

        var_residual = np.var(residual)
        strength = max(0.0, 1.0 - var_residual / var_detrended)

        return float(strength)

    def _robustness_test(self, original: pd.Series, residual: pd.Series) -> float:
        """
        Test decomposition robustness.

        Combines residual normality test and outlier robustness assessment.
        """
        try:
            clean_residual = residual.dropna()
            if len(clean_residual) < 8:
                return 0.5

            # Residual normality test (Jarque-Bera)
            _, jb_pvalue = jarque_bera(clean_residual)
            normality_score = min(1.0, max(0.0, jb_pvalue))

            # Outlier robustness (MAD vs STD)
            std_residual = clean_residual.std()
            if std_residual == 0:
                return normality_score

            mad = np.median(np.abs(clean_residual - clean_residual.median()))
            mad_std_ratio = mad / (std_residual + 1e-10)

            # Ideal MAD/STD ratio for normal distribution ≈ 0.675
            robustness_score = 1.0 - abs(mad_std_ratio - 0.675) / 0.675

            # Combined score
            return float((normality_score + robustness_score) / 2)

        except Exception as e:
            logging.warning(f"Error in robustness test: {e}")
            return 0.5

    def _calculate_composite_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate composite quality score.

        Accounts for metric weights and normalizes result to [0, 1] range.
        """
        if not scores:
            return 0.0

        weighted_score = 0.0
        total_weight = 0.0

        for metric_name, score in scores.items():
            # Skip composite score itself
            if metric_name == "composite_score":
                continue

            # Get metric weight
            metric_config = None
            for config in QualityMetricConfig:
                if config.value == metric_name:
                    metric_config = config
                    break

            if metric_config and metric_config in self.metric_weights:
                weight = self.metric_weights[metric_config]

                if weight < 0:
                    # For metrics where lower = better, invert
                    norm_score = 1.0 / (1.0 + score)
                    weighted_score += abs(weight) * norm_score
                else:
                    weighted_score += weight * score

                total_weight += abs(weight)

        if total_weight == 0:
            return 0.5

        composite = weighted_score / total_weight
        return float(min(1.0, max(0.0, composite)))

    def get_quality_summary(self, scores: Dict[str, float]) -> str:
        """
        Get text description of quality.

        Args:
            scores: Dictionary with quality scores

        Returns:
            String with quality description
        """
        composite = scores.get("composite_score", 0.0)

        if composite >= 0.8:
            quality = "excellent"
        elif composite >= 0.6:
            quality = "good"
        elif composite >= 0.4:
            quality = "fair"
        elif composite >= 0.2:
            quality = "poor"
        else:
            quality = "very poor"

        summary = f"Quality: {quality} (score: {composite:.3f})"

        # Add details for key metrics
        if "seasonal_strength" in scores:
            seasonal = scores["seasonal_strength"]
            summary += f", seasonality: {seasonal:.3f}"

        if "trend_strength" in scores:
            trend = scores["trend_strength"]
            summary += f", trend: {trend:.3f}"

        return summary