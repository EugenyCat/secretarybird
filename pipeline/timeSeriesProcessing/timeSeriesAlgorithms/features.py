# -----------------------------------------------------------------------------
# Class for feature engineering and scaling
# FeatureEngineer (scaling and feature generation)
# -----------------------------------------------------------------------------

from pipeline.helpers.configs import FeatureConfig
import pandas as pd
import numpy as np
import logging
from numba import njit
import holidays


@njit
def rolling_slope(arr, window):
    n = arr.shape[0]
    result = np.empty(n)
    result[:] = np.nan
    mean_i = (window - 1) / 2.0
    denom = 0.0
    for j in range(window):
        denom += (j - mean_i) ** 2
    for i in range(window - 1, n):
        s = 0.0
        for j in range(window):
            s += (j - mean_i) * arr[i - window + 1 + j]
        result[i] = s / denom
    return result


class FeatureGenerator:
    def __init__(self, interval, scaler_type='robust', lag_features=5, target_column='Open'):
        self.interval = interval
        self.scaler_type = scaler_type
        self.lag_features = lag_features
        self.target_column = target_column
        self.scaler = None
        self.custom_inverse = None

    def scale_data(self, data: pd.Series, log_scaling_stats=False):
        logging.debug("Starting data scaling (scaler: %s).", self.scaler_type)
        initial_stats = {
            'min': data.min(),
            'max': data.max(),
            'mean': data.mean(),
            'std': data.std(),
            'median': data.median(),
            'IQR': data.quantile(0.75) - data.quantile(0.25)
        }
        logging.debug("Initial statistics: %s", initial_stats)
        reshaped_data = data.values.reshape(-1, 1)
        if self.scaler_type == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif self.scaler_type == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        elif self.scaler_type == 'quantile':
            from sklearn.preprocessing import QuantileTransformer
            scaler = QuantileTransformer(output_distribution='normal')
        elif self.scaler_type == 'log':
            scaled_data = np.log1p(reshaped_data)
            self.custom_inverse = lambda x: np.expm1(x)
            scaled_data = scaled_data.flatten()
            logging.debug("Log scaling applied.")
            if log_scaling_stats:
                log_stats = {
                    'min': scaled_data.min(),
                    'max': scaled_data.max(),
                    'mean': scaled_data.mean(),
                    'std': scaled_data.std()
                }
                logging.info(f"[LOG] Data after log scaling stats: {log_stats}")
            return scaled_data
        else:
            raise ValueError("Unsupported scaler type. Choose from: 'minmax', 'standard', 'robust', 'quantile', 'log'.")


        scaled_data = scaler.fit_transform(reshaped_data).flatten()
        if log_scaling_stats:
            after_stats = {
                'min': scaled_data.min(),
                'max': scaled_data.max(),
                'mean': scaled_data.mean(),
                'std': scaled_data.std()
            }
            logging.debug(f"Statistics after scaling: {after_stats}")
        self.scaler = scaler
        return scaled_data

    def inverse_scale(self, scaled_data: np.ndarray):
        logging.debug("Starting inverse scaling.")
        if self.scaler_type == 'log' and self.custom_inverse is not None:
            reshaped_data = scaled_data.reshape(-1, 1)
            inversed = self.custom_inverse(reshaped_data)
            return inversed.flatten()
        else:
            reshaped_data = scaled_data.reshape(-1, 1)
            return self.scaler.inverse_transform(reshaped_data).flatten()

    def add_features(self, df: pd.DataFrame, windows=None, holiday_countries=None):
        """
        Generate features from the input DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame with time series data
            windows (list, optional): Custom window sizes. If None, uses configuration
            holiday_countries (list, optional): Custom holiday countries. If None, uses configuration

        Returns:
            pd.DataFrame: DataFrame with added features, including placeholders (-1)
                         for window features that exceed reliable calculation thresholds
        """
        logging.debug("Starting feature generation process.")
        df = df.copy()
        if not np.issubdtype(df.index.dtype, np.datetime64):
            df.index = pd.to_datetime(df.index, errors='coerce')

        base_col = f'{self.target_column}_scaled' if f'{self.target_column}_scaled' in df.columns else self.target_column
        features = df.copy()
        features[base_col] = df[base_col]

        # Use centralized configuration for lag features
        lags = FeatureConfig.get_lags(self.interval)
        lag_features = pd.concat([df[base_col].shift(lag).rename(f'lag_{lag}') for lag in lags], axis=1)
        features = pd.concat([features, lag_features], axis=1)

        # Use centralized configuration for windows or custom value
        if windows is None:
            windows = FeatureConfig.get_windows(self.interval)

        len_df = len(df)

        # Define window operations with their conditions
        window_operations = [
            ('roll_mean', lambda s, w: s.rolling(window=w, min_periods=1).mean(), None),
            ('roll_std', lambda s, w: s.rolling(window=w, min_periods=1).std(), None),
            ('roll_min', lambda s, w: s.rolling(window=w, min_periods=1).min(), None),
            ('roll_max', lambda s, w: s.rolling(window=w, min_periods=1).max(), None),
            ('roll_median', lambda s, w: s.rolling(window=w, min_periods=1).median(), None),
            ('roll_skew', lambda s, w: s.rolling(window=w, min_periods=1).skew(), lambda w: w > 2),
            ('roll_kurtosis', lambda s, w: s.rolling(window=w, min_periods=1).kurt(), lambda w: w > 3),
            ('rolling_trend', lambda s, w: rolling_slope(s.values, w), None)
        ]

        # Calculate all window-based features with a single loop
        for window in windows:
            is_reliable = window / len_df <= 0.15

            if not is_reliable:
                logging.debug(f"Window size {window} exceeds 15% of data length. Using placeholder values.")

            for op_name, op_func, op_condition in window_operations:
                # Skip if operation has a condition and it's not met
                if op_condition and not op_condition(window):
                    continue

                feature_name = f"{op_name}_{window}"

                # Use placeholder value (-1) or calculate feature based on reliability
                if not is_reliable:
                    features[feature_name] = -1
                else:
                    # For rolling_trend, use self.target_column instead of base_col
                    source_col = self.target_column if op_name == 'rolling_trend' else base_col
                    features[feature_name] = op_func(df[source_col], window)

        # Calculate derivative features
        features['first_derivative'] = df[self.target_column].diff()
        features['second_derivative'] = df[self.target_column].diff(2)

        # Calculate return and volatility features
        features['returns'] = df[self.target_column].pct_change()
        vol_window = windows[len(windows) // 2] if windows else 7
        features['volatility'] = features['returns'].rolling(window=vol_window, min_periods=1).std()

        # Generate holiday features using centralized configuration
        if holiday_countries is None:
            holiday_countries = FeatureConfig.get_holiday_countries()

        try:
            years = pd.to_datetime(df.index).year.unique().tolist()
            for country in holiday_countries:
                country_holidays = holidays.CountryHoliday(country, years=years, expand=True, observed=True)
                features[f'{country}_is_holiday'] = pd.Series(df.index.map(lambda d: d.date() in country_holidays),
                                                              index=df.index)
        except Exception as e:
            logging.warning(f"Failed to calculate holiday features: {e}")
            for country in holiday_countries:
                features[f'{country}_is_holiday'] = False

        logging.debug(f"Feature generation complete. Generated {len(features.columns)} features.")

        # Remove all rows that contain at least one NaN value
        features = features.dropna()  # todo: run experiment: do not remove Null but replace with -1 or another value
        return features