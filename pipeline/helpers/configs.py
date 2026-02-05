"""
Configuration classes for the pipeline system:

PropertySource: Enum representing different sources of time series properties (database, calculated, cached, or default).
PropertyGroupConfig: Dataclass defining configuration for property groups with fields, max age, and JSON field specifications.
FeatureConfig: Class containing configurations for time series feature engineering including lag features and rolling window sizes.
TimeSeriesPreprocessingConfig: Class containing configurations for time series preprocessing including max age settings for different intervals.


"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd


class InstrumentTypeConfig(Enum):
    """Financial instrument types"""

    CRYPTO = "crypto"  # Cryptocurrencies

    # For the future implementation
    """
    STOCK = "stock"  # Stocks
    BOND = "bond"  # Bonds
    FUTURES = "futures"  # Futures
    OPTIONS = "options"  # Options
    FOREX = "forex"  # Currency pairs
    COMMODITY = "commodity"  # Commodities
    INDEX = "index"  # Indices
    ETF = "etf"  # ETF funds
    """


class PropertySourceConfig(Enum):
    """Time series property source"""

    DATABASE = "database"
    CALCULATED = "calculated"
    CACHED = "cached"
    DEFAULT = "default"


class FeatureConfig:
    """
    Centralized configuration for time series feature engineering
    """

    # Configuration for lag features by interval
    LAG_CONFIG = {
        "1h": [1, 12, 24, 168],
        "12h": [1, 2, 3, 14],
        "1d": [1, 7, 14, 30],
        "3d": [1, 2, 7],
        "1w": [1, 2, 4],
        "1M": [1, 3, 6, 12],
    }

    # Configuration for rolling window sizes by interval
    WINDOW_CONFIG = {
        "1h": [3, 6, 12, 24],
        "12h": [2, 4, 6, 14],
        "1d": [3, 7, 14, 30],
        "3d": [2, 4, 7],
        "1w": [2, 4, 6],
        "1M": [3, 6, 12],
    }

    # Default holiday countries to include
    HOLIDAY_COUNTRIES = ["US"]

    @classmethod
    def get_lags(cls, interval):
        """
        Get lag values for a specific interval.

        Args:
            interval (str): Time interval code ('1h', '12h', '1d', etc.)

        Returns:
            list: List of lag values appropriate for the interval
        """
        try:
            return cls.LAG_CONFIG[interval]
        except KeyError:
            return [1, 2, 3, 4, 5]

    @classmethod
    def get_windows(cls, interval):
        """
        Get window sizes for a specific interval.

        Args:
            interval (str): Time interval code ('1h', '12h', '1d', etc.)

        Returns:
            list: List of window sizes appropriate for the interval
        """
        try:
            return cls.WINDOW_CONFIG[interval]
        except KeyError:
            return [3, 5, 7]

    @classmethod
    def get_holiday_countries(cls):
        """
        Get list of countries for holiday detection.

        Returns:
            list: List of country codes for holiday detection
        """
        return cls.HOLIDAY_COUNTRIES

    @classmethod
    def get_feature_schema_columns(cls, interval):
        """
        Get complete list of schema columns for features of a specified interval.
        Used for database schema creation and validation.

        Args:
            interval (str): Time interval code ('1h', '12h', '1d', etc.)

        Returns:
            list: List of tuples (column_name, data_type) for all features
        """
        columns = []

        # Add lag feature columns
        lags = cls.get_lags(interval)
        for lag in lags:
            columns.append((f"lag_{lag}", "Float64"))

        # Add window feature columns
        windows = cls.get_windows(interval)
        for window in windows:
            columns.append((f"roll_mean_{window}", "Float64"))
            columns.append((f"roll_std_{window}", "Float64"))
            columns.append((f"roll_min_{window}", "Float64"))
            columns.append((f"roll_max_{window}", "Float64"))
            columns.append((f"roll_median_{window}", "Float64"))

            # Conditionally add skew and kurtosis for appropriate window sizes
            if window > 2:
                columns.append((f"roll_skew_{window}", "Float64"))
            if window > 3:
                columns.append((f"roll_kurtosis_{window}", "Float64"))

            columns.append((f"rolling_trend_{window}", "Float64"))

        # Add derivative features
        columns.extend(
            [
                ("first_derivative", "Float64"),
                ("second_derivative", "Float64"),
                ("returns", "Float64"),
                ("volatility", "Float64"),
            ]
        )

        # Add holiday features
        for country in cls.get_holiday_countries():
            columns.append((f"{country}_is_holiday", "UInt8"))  # Boolean

        return columns


### Decomposition


class QualityMetricConfig(Enum):
    """
    Quality metrics:
    - for decomposition

    """

    MSE = "mse"
    MAE = "mae"
    AIC = "aic"
    BIC = "bic"
    RESIDUAL_AUTOCORR = "residual_autocorr"
    SILHOUETTE = "silhouette"
    ROBUSTNESS = "robustness"
    SEASONAL_STRENGTH = "seasonal_strength"
    TREND_STRENGTH = "trend_strength"


class DecompositionMethodConfig(Enum):
    """Time series decomposition methods"""

    STL = "stl"
    MSTL = "mstl"  # Multiple STL
    ROBUST_STL = "robust_stl"
    SSA = "ssa"
    FOURIER = "fourier"
    PROPHET = "prophet"
    TBATS = "tbats"
    NBEATS = "nbeats"  # Neural approach
    AUTO = "auto"
    # EMD = "emd"  # Empirical Mode Decomposition # not implemented
    # WAVELET = "wavelet"
    # ENSEMBLE = "ensemble"


@dataclass
class DecompositionConfig:
    """Decomposition configuration"""

    method: DecompositionMethodConfig = DecompositionMethodConfig.AUTO
    period: Optional[int] = None
    secondary_periods: Optional[List[int]] = None
    robust: bool = True
    quality_metrics: List[QualityMetricConfig] = None
    force_positive: bool = False
    interval_specific_params: Optional[Dict[str, Any]] = None


@dataclass
class DecompositionResultConfig:
    """Decomposition result"""

    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    method: DecompositionMethodConfig
    quality_scores: Dict[str, float]
    metadata: Dict[str, Any]
    period_used: Optional[int] = None
    secondary_periods: Optional[List[int]] = None