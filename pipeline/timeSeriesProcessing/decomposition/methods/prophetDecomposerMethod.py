"""
Time series decomposition method based on Prophet.

Implements Prophet method from Facebook for time series with strong trends
and structural breaks. Uses piecewise linear regression for trend
and additive/multiplicative seasonal components.

Optimal for conditions: trend_strength > 0.25 AND missing_ratio > 0.02 AND crypto=True
according to Enhanced Decision Tree.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from prophet import Prophet

from pipeline.helpers.utils import validate_required_locals
from pipeline.timeSeriesProcessing.decomposition.methods.baseDecomposerMethod import (
    BaseDecomposerMethod,
)

__version__ = "1.0.0"


class ProphetDecomposerMethod(BaseDecomposerMethod):
    """
    Prophet decomposition method from Facebook.

    Specially designed for time series with strong seasonality,
    holiday effects and structural trend breaks.
    Especially effective for financial and business data.

    Applies "Fail Fast" principle for critical dependencies.
    Optimal for crypto data with trend_strength > 0.25.
    """

    DEFAULT_CONFIG = {
        **BaseDecomposerMethod.DEFAULT_CONFIG,
        # Operation mode
        "decomposition_mode": True,  # Optimization for decomposition vs forecasting (Does not require adaptation)
        "force_fast_decomposition": True,  # Forced performance optimization (Does not require adaptation)
        # Constants
        "changepoint_range": 0.8,  # Data fraction for changepoint search (Does not require adaptation)
        "interval_width": 0.80,  # Interval width (Does not require adaptation)
        "uncertainty_samples": 0,  # Disabled for decomposition (Does not require adaptation)
        "mcmc_samples": 0,  # Forcibly disabled for performance (Does not require adaptation)
        "custom_seasonalities": [],  # List of custom seasonalities (Does not require adaptation)
        "growth": "linear",  # linear/logistic/flat (Does not require adaptation)
        "add_country_holidays": False,  # Add national holidays (Does not require adaptation)
        "country_name": "US",  # Country for holidays  (Does not require adaptation)
        "holidays_prior_scale": 10.0,  # Holiday strength  (Does not require adaptation)
        # Prophet seasonalities
        # "yearly_seasonality": "auto",  # [AUTO] auto/True/False (adapted in configDecomposition)
        # "weekly_seasonality": "auto",  # [AUTO] auto/True/False (adapted in configDecomposition)
        # "daily_seasonality": "auto",  # [AUTO] auto/True/False (adapted in configDecomposition)
        # "seasonality_mode": "additive",  # additive/multiplicative (adapted in configDecomposition)
        # Changepoint parameters
        # "changepoint_prior_scale": 0.05,  # Changepoint flexibility (adapted in configDecomposition)
        # "n_changepoints": 25,  # Number of potential changepoints (adapted in configDecomposition)
        # "seasonality_prior_scale": 10.0,  # Seasonality strength (adapted in configDecomposition)
        # TODO not urgent: Improve using these parameters (not used in current implementation):
        # "add_custom_seasonality": True,   # should be adapted in configDecomposition
        # "custom_fourier_order": None,     # should be adapted in configDecomposition
        # "min_data_length": 30,            # should be adapted in configDecomposition
        # "optimal_data_length": 365,       # should be adapted in configDecomposition
        # "confidence_threshold": 0.90,     # should be adapted in configDecomposition
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Prophet method.

        Args:
            config: Configuration with parameters (required)

        Raises:
            ValueError: If Prophet is unavailable or configuration is invalid
        """

        # Initialize base class
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged_config)

        # Extract Prophet-specific parameters
        self.decomposition_mode = self.config["decomposition_mode"]
        self.force_fast_decomposition = self.config["force_fast_decomposition"]
        self.yearly_seasonality = self.config["yearly_seasonality"]
        self.weekly_seasonality = self.config["weekly_seasonality"]
        self.daily_seasonality = self.config["daily_seasonality"]
        self.seasonality_mode = self.config["seasonality_mode"]
        self.changepoint_prior_scale = self.config["changepoint_prior_scale"]
        self.n_changepoints = self.config["n_changepoints"]
        self.changepoint_range = self.config["changepoint_range"]
        self.seasonality_prior_scale = self.config["seasonality_prior_scale"]
        self.holidays_prior_scale = self.config["holidays_prior_scale"]
        self.interval_width = self.config["interval_width"]
        self.uncertainty_samples = self.config["uncertainty_samples"]
        self.mcmc_samples = self.config["mcmc_samples"]
        self.custom_seasonalities = self.config["custom_seasonalities"]
        self.growth = self.config["growth"]
        self.add_country_holidays = self.config["add_country_holidays"]
        self.country_name = self.config["country_name"]

        logging.info(f"{self} initialized with mode={self.seasonality_mode}")

    def __str__(self) -> str:
        """String representation for logging."""
        return (
            f"ProphetDecomposerMethod(mode={self.seasonality_mode}, "
            f"changepoint_scale={self.changepoint_prior_scale}, "
            f"growth={self.growth})"
        )

    def _validate_config(self) -> None:
        """Validate method configuration (Fail Fast)."""

        # Validate required Prophet parameters
        required_params = [
            "yearly_seasonality",
            "weekly_seasonality",
            "daily_seasonality",
            "seasonality_mode",
            "changepoint_prior_scale",
            "seasonality_prior_scale",
        ]

        validate_required_locals(required_params, self.config)

        # Validate parameter values
        if self.config["seasonality_mode"] not in ["additive", "multiplicative"]:
            raise ValueError(
                f"Invalid seasonality_mode: {self.config['seasonality_mode']}. "
                f"Valid values: 'additive', 'multiplicative'"
            )

        if not (0 < self.config["changepoint_prior_scale"] <= 5.0):
            raise ValueError(
                f"changepoint_prior_scale must be in range (0, 5.0], "
                f"got: {self.config['changepoint_prior_scale']}"
            )

        if not (0 < self.config["interval_width"] < 1.0):
            raise ValueError(
                f"interval_width must be in range (0, 1), "
                f"got: {self.config['interval_width']}"
            )

        if self.config["growth"] not in ["linear", "logistic", "flat"]:
            raise ValueError(
                f"Invalid growth: {self.config['growth']}. "
                f"Valid values: 'linear', 'logistic', 'flat'"
            )

        if self.config["n_changepoints"] < 0:
            raise ValueError(
                f"n_changepoints must be >= 0, got: {self.config['n_changepoints']}"
            )

        if not (0 < self.config["changepoint_range"] <= 1.0):
            raise ValueError(
                f"changepoint_range must be in range (0, 1], "
                f"got: {self.config['changepoint_range']}"
            )

    def process(
        self, data: pd.Series, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform decomposition using Prophet method.

        Args:
            data: Time series for decomposition
            context: Processing context (required)

        Returns:
            Standardized decomposition result
        """
        try:
            # 1. CRITICAL fail-fast validation
            critical_validation = self.validate_input_critical(data)
            if critical_validation is not None:
                return critical_validation

            # 2. Standard input data validation (base method)
            validation = self.validate_input(data)
            if validation["status"] == "error":
                return validation

            # 2. Context validation (Fail Fast)
            if not context:
                return self.handle_error(
                    ValueError("Context is required for Prophet decomposition"),
                    "context_validation",
                )

            # 3. Extract context (base method)
            context_params = self.extract_context_parameters(context)

            # 4. Validate context dependencies (Fail Fast)
            self._validate_context_dependencies(context)

            # 5. Log start (base method)
            self.log_analysis_start(data, context_params)

            # 6. Data preprocessing (base method)
            processed_data = self.preprocess_data(data)

            # 7. Prepare data for Prophet with missing data analysis
            missing_data_info = self._analyze_missing_data(
                processed_data, context_params
            )
            prophet_df = self._prepare_prophet_dataframe(
                processed_data, missing_data_info
            )

            # 8. Create and configure Prophet model
            model = self._create_prophet_model(context)

            # 9. Simplified Bayesian logic for decomposition mode
            try:
                # Prophet optimization: MAP estimation only (no MCMC sampling)
                if self.decomposition_mode or self.force_fast_decomposition:
                    logging.debug(
                        f"{self} - Prophet decomposition mode: using MAP estimation only "
                        f"(mcmc_samples=0, uncertainty_samples=0)"
                    )
                    # MAP (Maximum A Posteriori) estimation without Bayesian posterior sampling
                    model.fit(prophet_df)
                elif self.mcmc_samples > 0:
                    # Bayesian posterior sampling only if explicitly not in decomposition mode
                    logging.debug(
                        f"{self} - Prophet forecasting mode: using MCMC sampling "
                        f"(mcmc_samples={self.mcmc_samples})"
                    )
                    model.fit(prophet_df)
                else:
                    # Default MAP estimation
                    model.fit(prophet_df)
            except Exception as e:
                return self.handle_error(
                    RuntimeError(f"Prophet model training error: {str(e)}"),
                    "model_fitting",
                )

            # 10. Get forecast (for decomposition we use original data)
            try:
                forecast = model.predict(prophet_df)
            except Exception as e:
                return self.handle_error(
                    RuntimeError(f"Prophet forecast error: {str(e)}"),
                    "prediction",
                )

            # 11. Extract decomposition components
            decomposition_result = self._extract_components(
                processed_data, forecast, model
            )
            trend, seasonal, residual, components_info = decomposition_result

            # 12. Validate decomposition results
            validation_result = self._validate_decomposition_results(
                trend, seasonal, residual
            )
            if validation_result["status"] == "error":
                return validation_result

            # 13. Prepare additional data
            additional_data = self._prepare_additional_data(
                model, forecast, components_info
            )

            # 14. Add missing data information for Bayesian analysis
            additional_data["missing_data_info"] = missing_data_info

            # 15. Create standardized result (base method)
            return self.prepare_decomposition_result(
                trend,
                seasonal,
                residual,
                processed_data,
                context_params,
                additional_data,
            )

        except Exception as e:
            return self.handle_error(e, "Prophet decomposition")

    def _validate_context_dependencies(self, context: Dict[str, Any]) -> None:
        """
        Validate context dependencies (Fail Fast).

        Args:
            context: Processing context

        Raises:
            ValueError: If critical dependencies are missing
        """
        # Check context structure
        if "currentProperties" not in context:
            raise ValueError(
                "Missing 'currentProperties' in context. "
                "Ensure previous processors were executed correctly."
            )

        current_props = context["currentProperties"]

        # Check for required property groups
        required_groups = ["analyzer", "periodicity"]
        missing_groups = [
            group for group in required_groups if group not in current_props
        ]
        if missing_groups:
            raise ValueError(
                f"Missing required property groups: {missing_groups}. "
                f"Prophet requires prior analysis and periodicity detection."
            )

        # Check critical analyzer properties
        analyzer_props = current_props["analyzer"]
        required_analyzer_props = [
            "volatility",
            "estimated_trend_strength",
            "noise_level",
            "missing_ratio",
        ]
        missing_analyzer = [
            prop for prop in required_analyzer_props if prop not in analyzer_props
        ]
        if missing_analyzer:
            raise ValueError(
                f"Missing critical analyzer properties: {missing_analyzer}"
            )

        # Check periodicity properties
        periodicity_props = current_props["periodicity"]
        if "main_period" not in periodicity_props:
            raise ValueError(
                "Missing required property 'main_period' in periodicity"
            )

    def _prepare_prophet_dataframe(
        self, data: pd.Series, missing_info: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Prepare DataFrame in Prophet format.

        Args:
            data: Time series
            missing_info: Missing data information for Bayesian processing

        Returns:
            DataFrame with 'ds' and 'y' columns, ready for Prophet

        Raises:
            ValueError: If data cannot be converted to Prophet format
        """
        try:
            df = pd.DataFrame({"ds": data.index, "y": data.values})

            # Prophet requires datetime index
            if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
                try:
                    df["ds"] = pd.to_datetime(df["ds"])
                    logging.debug(f"{self} - Index converted to datetime format")
                except Exception as conversion_error:
                    # Create artificial dates as fallback
                    df["ds"] = pd.date_range(
                        start="2020-01-01", periods=len(data), freq="D"
                    )
                    logging.warning(
                        f"{self} - Failed to convert index to datetime ({conversion_error}), "
                        f"using artificial dates"
                    )

            # Handle missing data for Bayesian posterior sampling
            if missing_info and missing_info.get("has_missing", False):
                logging.info(
                    f"Prophet DataFrame: handling {missing_info['missing_count']} missing values "
                    f"using strategy '{missing_info['bayesian_strategy']}'"
                )

                # Prophet can work with NaN, they will be processed through posterior sampling
                # However if too many missing values, apply preliminary interpolation
                if missing_info["missing_ratio"] > 0.3:  # More than 30% missing
                    logging.warning(
                        f"Prophet: high missing ratio ({missing_info['missing_ratio']:.1%}), "
                        f"applying preliminary interpolation"
                    )
                    df["y"] = df["y"].interpolate(method="time").bfill().ffill()
            else:
                # Regular validation for cases without missing data
                if df.isna().any().any():
                    logging.warning(
                        "Prophet DataFrame: NaN detected, applying fill"
                    )
                    df["y"] = df["y"].bfill().ffill()

            return df

        except Exception as e:
            raise ValueError(f"Error preparing data for Prophet: {str(e)}")

    def _create_prophet_model(self, context: Dict[str, Any]) -> Prophet:
        """
        Create and configure Prophet model.

        Args:
            context: Processing context

        Returns:
            Configured Prophet model

        Raises:
            ValueError: If model cannot be created
        """
        try:
            # Base Prophet parameters
            model_params = {
                "yearly_seasonality": self.yearly_seasonality,
                "weekly_seasonality": self.weekly_seasonality,
                "daily_seasonality": self.daily_seasonality,
                "seasonality_mode": self.seasonality_mode,
                "changepoint_prior_scale": self.changepoint_prior_scale,
                "n_changepoints": self.n_changepoints,
                "changepoint_range": self.changepoint_range,
                "seasonality_prior_scale": self.seasonality_prior_scale,
                "holidays_prior_scale": self.holidays_prior_scale,
                "interval_width": self.interval_width,
                "uncertainty_samples": self.uncertainty_samples,
                "growth": self.growth,
            }

            # Simplified Bayesian logic for decomposition
            if self.decomposition_mode or self.force_fast_decomposition:
                # Prophet optimization: complete disable of Bayesian sampling for decomposition
                # MAP estimation instead of full posterior sampling
                logging.debug(
                    f"{self} - Simplified Bayesian mode: MCMC disabled, uncertainty disabled"
                )
                # mcmc_samples not added to model_params (remains default=0)
                pass
            elif self.mcmc_samples > 0:
                # Full Bayesian mode for forecasting
                model_params["mcmc_samples"] = self.mcmc_samples
                logging.debug(
                    f"{self} - Full Bayesian mode: mcmc_samples={self.mcmc_samples}"
                )

            # Create model
            model = Prophet(**model_params)

            # Add custom seasonality based on detected period
            self._add_custom_seasonality(model, context)

            # Add national holidays
            if self.add_country_holidays:
                try:
                    model.add_country_holidays(country_name=self.country_name)
                    logging.info(
                        f"{self} - Added country holidays: {self.country_name}"
                    )
                except Exception as e:
                    logging.warning(f"{self} - Failed to add holidays: {e}")

            # Add custom seasonalities
            for seasonality in self.custom_seasonalities:
                try:
                    model.add_seasonality(**seasonality)
                    logging.info(
                        f"{self} - Added custom seasonality: {seasonality['name']}"
                    )
                except Exception as e:
                    logging.warning(
                        f"{self} - Error adding seasonality {seasonality}: {e}"
                    )

            return model

        except Exception as e:
            raise ValueError(f"Error creating Prophet model: {str(e)}")

    def _add_custom_seasonality(self, model: Prophet, context: Dict[str, Any]) -> None:
        """
        Add custom seasonality based on detected period.

        Args:
            model: Prophet model
            context: Processing context
        """
        try:
            # Extract periodicity from context
            periodicity_props = context["currentProperties"]["periodicity"]
            main_period = periodicity_props.get("main_period")

            # Add custom seasonality only if period is significant
            if main_period and main_period > 1:
                # Exclude standard Prophet periods
                standard_periods = [7, 365.25]  # weekly, yearly

                if main_period not in standard_periods:
                    # Adaptive fourier_order determination based on period
                    fourier_order = min(10, max(3, main_period // 4))

                    # Determine mode based on overall seasonality_mode
                    seasonality_config = {
                        "name": "custom_period",
                        "period": main_period,
                        "fourier_order": fourier_order,
                        "mode": self.seasonality_mode,
                    }

                    model.add_seasonality(**seasonality_config)
                    logging.info(
                        f"{self} - Added custom seasonality: period={main_period}, "
                        f"fourier_order={fourier_order}, mode={self.seasonality_mode}"
                    )

        except (KeyError, TypeError, ValueError) as e:
            # Continue without custom seasonality, but log warning
            logging.warning(f"{self} - Failed to add custom seasonality: {e}")

    def _extract_components(
        self, original_data: pd.Series, forecast: pd.DataFrame, model: Prophet
    ) -> tuple:
        """
        Extract decomposition components from Prophet forecast.

        Args:
            original_data: Original time series
            forecast: Prophet forecast
            model: Trained Prophet model

        Returns:
            Tuple[trend, seasonality, residuals, component info]

        Raises:
            ValueError: If components cannot be extracted
        """
        try:
            # Validate input data
            if len(forecast) != len(original_data):
                raise ValueError(
                    f"Forecast dimension ({len(forecast)}) does not match "
                    f"original data dimension ({len(original_data)})"
                )

            # Extract trend
            if "trend" not in forecast.columns:
                raise ValueError("Missing 'trend' component in Prophet forecast")

            trend = pd.Series(forecast["trend"].values, index=original_data.index)

            # Collect all seasonal components
            seasonal_components = []
            seasonalities_info = []

            # Vectorized collection of standard seasonalities
            standard_seasonalities = ["yearly", "weekly", "daily"]
            available_standard = [
                s for s in standard_seasonalities if s in forecast.columns
            ]

            if available_standard:
                # Vectorized extraction via pandas slices
                standard_data = forecast[
                    available_standard
                ].values.T  # Transpose for list comprehension
                seasonal_components.extend(standard_data)
                seasonalities_info.extend(available_standard)

            # Vectorized collection of custom seasonalities
            custom_seasonalities = [
                s
                for s in model.seasonalities
                if s not in standard_seasonalities and s in forecast.columns
            ]

            if custom_seasonalities:
                # Vectorized extraction of custom
                custom_data = forecast[
                    custom_seasonalities
                ].values.T  # Transpose for list comprehension
                seasonal_components.extend(custom_data)
                seasonalities_info.extend(custom_seasonalities)

            # Holidays as part of seasonality
            if "holidays" in forecast.columns:
                seasonal_components.append(forecast["holidays"].values)
                seasonalities_info.append("holidays")

            # Sum seasonal components
            if seasonal_components:
                if self.seasonality_mode == "multiplicative":
                    # For multiplicative model, correctly combine
                    seasonal_array = np.ones(len(original_data))
                    for comp in seasonal_components:
                        seasonal_array *= 1 + comp
                    seasonal_array -= 1  # Return to additive form for consistency
                else:
                    # Additive model
                    seasonal_array = np.sum(seasonal_components, axis=0)

                seasonal = pd.Series(seasonal_array, index=original_data.index)
            else:
                seasonal = pd.Series(
                    np.zeros(len(original_data)), index=original_data.index
                )
                logging.warning(f"{self} - No seasonal components found in forecast")

            # Calculate residuals
            if "yhat" not in forecast.columns:
                raise ValueError("Missing forecast 'yhat' in Prophet results")

            yhat = pd.Series(forecast["yhat"].values, index=original_data.index)
            residual = original_data - yhat

            # Component information
            components_info = {
                "seasonalities": seasonalities_info,
                "has_holidays": "holidays" in forecast.columns,
                "has_changepoints": hasattr(model, "changepoints")
                and len(model.changepoints) > 0,
                "n_changepoints": (
                    len(model.changepoints) if hasattr(model, "changepoints") else 0
                ),
                "forecast_quality": self._assess_forecast_quality(original_data, yhat),
                "seasonality_mode": self.seasonality_mode,
            }

            logging.debug(
                f"{self} - Extracted components: trend, seasonal ({len(seasonalities_info)} types), residual"
            )

            return trend, seasonal, residual, components_info

        except Exception as e:
            raise ValueError(f"Error extracting decomposition components: {str(e)}")

    def _validate_decomposition_results(
        self, trend: pd.Series, seasonal: pd.Series, residual: pd.Series
    ) -> Dict[str, Any]:
        """
        Validate decomposition results.

        Args:
            trend: Trend component
            seasonal: Seasonal component
            residual: Residual component

        Returns:
            Dict with validation result
        """
        try:
            # Check for NaN/Inf
            for name, component in [
                ("trend", trend),
                ("seasonal", seasonal),
                ("residual", residual),
            ]:
                if component.isna().any():
                    return {
                        "status": "error",
                        "message": f"Component '{name}' contains NaN values",
                    }
                if np.isinf(component).any():
                    return {
                        "status": "error",
                        "message": f"Component '{name}' contains infinite values",
                    }

            # Check size consistency
            lengths = [len(trend), len(seasonal), len(residual)]
            if len(set(lengths)) > 1:
                return {
                    "status": "error",
                    "message": f"Components have different sizes: {lengths}",
                }

            return {"status": "success"}

        except Exception as e:
            return {
                "status": "error",
                "message": f"Error validating results: {str(e)}",
            }

    def _assess_forecast_quality(
        self, original: pd.Series, forecast: pd.Series
    ) -> Dict[str, float]:
        """
        Assess Prophet forecast quality.

        Args:
            original: Original data
            forecast: Forecast data

        Returns:
            Dict with quality metrics
        """
        try:
            # Use universal QualityEvaluator for MSE/MAE/MAPE
            quality_evaluator = QualityEvaluator(evaluation_type="prediction")
            prediction_metrics = quality_evaluator.evaluate_prediction(
                original, forecast, ["mse", "mae", "mape", "r2"]
            )

            return {
                "mse": prediction_metrics["mse"],
                "mae": prediction_metrics["mae"],
                "r2": prediction_metrics["r2"],
                "mape": prediction_metrics["mape"],
            }
        except Exception:
            return {"mse": float("inf"), "mae": float("inf"), "r2": 0.0, "mape": 100.0}

    def _prepare_additional_data(
        self, model: Prophet, forecast: pd.DataFrame, components_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare additional data for result.

        Args:
            model: Trained Prophet model
            forecast: Prophet forecast
            components_info: Component information

        Returns:
            Dict with additional data
        """
        additional_data = {
            # Main model parameters
            "model_type": "prophet",
            "seasonality_mode": self.seasonality_mode,
            "growth": self.growth,
            # Changepoint information
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "n_changepoints_detected": components_info.get("n_changepoints", 0),
            "changepoints_detected": hasattr(model, "changepoints")
            and len(model.changepoints) > 0,
            # Seasonality information
            "seasonalities_fitted": components_info.get("seasonalities", []),
            "n_seasonalities": len(components_info.get("seasonalities", [])),
            "has_holidays": components_info.get("has_holidays", False),
            # Model parameters
            "prophet_params": {
                "changepoint_prior_scale": self.changepoint_prior_scale,
                "seasonality_prior_scale": self.seasonality_prior_scale,
                "holidays_prior_scale": self.holidays_prior_scale,
                "interval_width": self.interval_width,
                "uncertainty_samples": self.uncertainty_samples,
                "mcmc_samples": self.mcmc_samples,
            },
            # Forecast quality
            "forecast_quality": components_info.get("forecast_quality", {}),
            # Uncertainty
            "has_uncertainty": "yhat_lower" in forecast.columns
            and "yhat_upper" in forecast.columns,
            "uncertainty_width": self.interval_width,
            # Bayesian sampling information
            "bayesian_sampling": self.mcmc_samples > 0,
            # Prophet version
            "prophet_version": Prophet().__class__.__module__,
        }

        # Add changepoint dates if available
        if hasattr(model, "changepoints") and len(model.changepoints) > 0:
            # Limit quantity for performance
            changepoint_dates = (
                model.changepoints[:10].tolist()
                if len(model.changepoints) > 10
                else model.changepoints.tolist()
            )
            additional_data["changepoint_dates"] = [
                str(date) for date in changepoint_dates
            ]

            # Analyze piecewise linear regression and changepoint quality
            trend_analysis = self._analyze_piecewise_trend(model, forecast)
            additional_data.update(trend_analysis)

        return additional_data

    def _analyze_missing_data(
        self, data: pd.Series, context_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze missing data with optimization for decomposition.

        For decomposition mode, simplifies missing value handling strategy,
        avoiding excessive Bayesian inference to improve performance.

        Args:
            data: Time series for analysis
            context_params: Context parameters

        Returns:
            Missing data information and handling strategy
        """
        try:
            missing_info = {
                "has_missing": data.isna().any(),
                "missing_count": int(data.isna().sum()),
                "missing_ratio": float(data.isna().sum() / len(data)),
                "missing_pattern": "none",
                "bayesian_strategy": "none",
                "mcmc_recommended": False,
            }

            if missing_info["has_missing"]:
                # For decomposition mode use simplified strategy
                if self.decomposition_mode or self.force_fast_decomposition:
                    # Forcibly disable MCMC for decomposition
                    missing_info["bayesian_strategy"] = "interpolation"
                    missing_info["mcmc_recommended"] = False
                    missing_info["missing_pattern"] = "decomposition_mode"

                    logging.info(
                        f"Prophet decomposition mode: {missing_info['missing_count']} missing values "
                        f"({missing_info['missing_ratio']:.1%}), strategy: fast_interpolation"
                    )
                else:
                    # Regular logic only for forecasting mode
                    missing_mask = data.isna()

                    # Simplified classification without excessive Bayesian strategies
                    if missing_mask.sum() > len(data) * 0.1:  # More than 10% missing
                        missing_info["missing_pattern"] = "high_density"
                        missing_info["bayesian_strategy"] = "interpolation"  # Not MCMC
                    elif missing_mask.sum() > len(data) * 0.02:  # 2-10% missing
                        missing_info["missing_pattern"] = "moderate"
                        missing_info["bayesian_strategy"] = "interpolation"
                    else:
                        missing_info["missing_pattern"] = "sparse"
                        missing_info["bayesian_strategy"] = "interpolation"

                    missing_info["mcmc_recommended"] = (
                        False  # Disabled for decomposition
                    )

                    logging.info(
                        f"Prophet missing data: {missing_info['missing_count']} missing values "
                        f"({missing_info['missing_ratio']:.1%}), strategy: {missing_info['bayesian_strategy']}"
                    )
            else:
                logging.debug("Prophet: no missing data detected")

            return missing_info

        except Exception as e:
            logging.warning(f"Prophet missing data analysis failed: {e}")
            return {
                "has_missing": False,
                "missing_count": 0,
                "missing_ratio": 0.0,
                "missing_pattern": "none",
                "bayesian_strategy": "none",
                "mcmc_recommended": False,
            }

    def _analyze_piecewise_trend(
        self, model: Prophet, forecast: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze piecewise linear regression for changepoints.

        Extracts information about trend resegmentation quality,
        model flexibility and changepoint statistical characteristics.

        Args:
            model: Trained Prophet model
            forecast: Forecast result

        Returns:
            Dictionary with piecewise linear regression metrics
        """
        try:
            analysis = {}

            # Extract trend component
            if "trend" in forecast.columns:
                trend = forecast["trend"].values

                # Analyze trend flexibility (variability)
                trend_diff = np.diff(trend)
                trend_diff2 = np.diff(trend_diff)  # Second derivative

                analysis["trend_flexibility"] = (
                    float(np.std(trend_diff2)) if len(trend_diff2) > 0 else 0.0
                )
                analysis["trend_smoothness"] = (
                    float(1.0 / (1.0 + np.std(trend_diff2)))
                    if len(trend_diff2) > 0
                    else 1.0
                )
                analysis["trend_variance"] = (
                    float(np.var(trend)) if len(trend) > 0 else 0.0
                )
                analysis["trend_mean_gradient"] = (
                    float(np.mean(trend_diff)) if len(trend_diff) > 0 else 0.0
                )

                # Analyze changepoint quality
                if hasattr(model, "changepoints") and len(model.changepoints) > 0:
                    n_changepoints = len(model.changepoints)
                    data_length = len(trend)

                    # Changepoint density
                    changepoint_density = (
                        n_changepoints / data_length if data_length > 0 else 0.0
                    )

                    # Changepoint effectiveness (average change at breakpoints)
                    changepoint_positions = np.linspace(
                        0, len(trend) - 1, n_changepoints, dtype=int
                    )
                    changepoint_effects = []

                    for i, pos in enumerate(changepoint_positions):
                        if (
                            pos > 5 and pos < len(trend) - 5
                        ):  # Enough data for analysis
                            before = np.mean(trend[max(0, pos - 5) : pos])
                            after = np.mean(trend[pos : min(len(trend), pos + 5)])
                            effect = abs(after - before)
                            changepoint_effects.append(effect)

                    analysis["changepoint_density"] = float(changepoint_density)
                    analysis["changepoint_avg_effect"] = (
                        float(np.mean(changepoint_effects))
                        if changepoint_effects
                        else 0.0
                    )
                    analysis["changepoint_max_effect"] = (
                        float(np.max(changepoint_effects))
                        if changepoint_effects
                        else 0.0
                    )
                    analysis["changepoint_effectiveness"] = (
                        float(np.std(changepoint_effects))
                        if len(changepoint_effects) > 1
                        else 0.0
                    )

                # Piecewise linear regression quality
                if len(trend) > 1:
                    # Overall R² for entire trend
                    trend_mean = np.mean(trend)
                    ss_tot = np.sum((trend - trend_mean) ** 2)

                    # Linear fit for comparison
                    x = np.arange(len(trend))
                    linear_coef = np.polyfit(x, trend, 1)
                    linear_pred = np.polyval(linear_coef, x)
                    ss_res_linear = np.sum((trend - linear_pred) ** 2)

                    # Improvement over simple linear regression
                    piecewise_improvement = (
                        1.0 - (ss_res_linear / ss_tot) if ss_tot > 0 else 0.0
                    )

                    analysis["piecewise_vs_linear_improvement"] = float(
                        piecewise_improvement
                    )
                    analysis["trend_linearity"] = float(
                        1.0 - piecewise_improvement
                    )  # Inverse metric
                    analysis["trend_complexity"] = (
                        float(n_changepoints / data_length)
                        if "n_changepoints" in locals()
                        else 0.0
                    )

            logging.debug(
                f"Prophet piecewise trend analysis: {len(analysis)} metrics extracted"
            )
            return analysis

        except Exception as e:
            logging.warning(f"Prophet piecewise trend analysis failed: {e}")
            return {
                "trend_flexibility": 0.0,
                "trend_smoothness": 1.0,
                "changepoint_density": 0.0,
                "piecewise_vs_linear_improvement": 0.0,
            }

    def require_context_property(
        self, context: Dict[str, Any], group: str, property_name: str
    ) -> Any:
        """
        Extract required property from context.

        Args:
            context: Context
            group: Property group
            property_name: Property name

        Returns:
            Property value

        Raises:
            ValueError: If property is missing
        """
        return context["currentProperties"][group][property_name]