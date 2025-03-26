
# -----------------------------------------------------------------------------
# Класс для инженерии признаков и масштабирования
# FeatureEngineer (масштабирование и генерация признаков)
# -----------------------------------------------------------------------------

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


class FeatureEngineer:
    def __init__(self, interval, scaler_type='robust', lag_features=5, target_column='Open'):
        self.interval = interval
        self.scaler_type = scaler_type
        self.lag_features = lag_features
        self.target_column = target_column
        self.scaler = None
        self.custom_inverse = None

    def scale_data(self, data: pd.Series, log_scaling_stats=False):
        logging.debug("Начало масштабирования данных (scaler: %s).", self.scaler_type)
        initial_stats = {
            'min': data.min(),
            'max': data.max(),
            'mean': data.mean(),
            'std': data.std(),
            'median': data.median(),
            'IQR': data.quantile(0.75) - data.quantile(0.25)
        }
        logging.debug("Начальные статистики: %s", initial_stats)
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
            logging.debug("Применено log масштабирование.")
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
            logging.debug(f"Статистики после масштабирования: {after_stats}")
        self.scaler = scaler
        return scaled_data

    def inverse_scale(self, scaled_data: np.ndarray):
        logging.debug("Начало обратного масштабирования.")
        if self.scaler_type == 'log' and self.custom_inverse is not None:
            reshaped_data = scaled_data.reshape(-1, 1)
            inversed = self.custom_inverse(reshaped_data)
            return inversed.flatten()
        else:
            reshaped_data = scaled_data.reshape(-1, 1)
            return self.scaler.inverse_transform(reshaped_data).flatten()

    def add_features(self, df: pd.DataFrame, windows=None, holiday_countries=['US']):
        logging.debug("Начало создания дополнительных признаков.")
        df = df.copy()
        if not np.issubdtype(df.index.dtype, np.datetime64):
            df.index = pd.to_datetime(df.index, errors='coerce')
        base_col = f'{self.target_column}_scaled' if f'{self.target_column}_scaled' in df.columns else self.target_column
        features = df.copy()
        features[base_col] = df[base_col]
        lag_config = {
            '1h': [1, 12, 24, 168],
            '12h': [1, 2, 3, 14],
            '1d': [1, 7, 14, 30],
            '3d': [1, 2, 7],
            '1w': [1, 2, 4],
            '1M': [1, 3, 6, 12]
        }
        lags = lag_config.get(self.interval, list(range(1, self.lag_features + 1)))
        lag_features = pd.concat([df[base_col].shift(lag).rename(f'lag_{lag}') for lag in lags], axis=1)
        features = pd.concat([features, lag_features], axis=1)
        default_windows = {
            '1h': [3, 6, 12, 24],
            '12h': [2, 4, 6, 14],
            '1d': [3, 7, 14, 30],
            '3d': [2, 4, 7],
            '1w': [2, 4, 6],
            '1M': [3, 6, 12]
        }
        if windows is None:
            windows = default_windows.get(self.interval, [3, 5, 7])
        len_df = len(df)
        for window in windows:
            # условие если размер окна не превышает 15% размера всех данных (то есть окно 12 месяцев только для массива где минимум есть данные за 80 месяцев)
            if window / len_df <= 0.15:
                continue
            features[f'roll_mean_{window}'] = df[base_col].rolling(window=window, min_periods=1).mean()
            features[f'roll_std_{window}'] = df[base_col].rolling(window=window, min_periods=1).std()
            features[f'roll_min_{window}'] = df[base_col].rolling(window=window, min_periods=1).min()
            features[f'roll_max_{window}'] = df[base_col].rolling(window=window, min_periods=1).max()
            features[f'roll_median_{window}'] = df[base_col].rolling(window=window, min_periods=1).median()
            if window > 2:
                features[f'roll_skew_{window}'] = df[base_col].rolling(window=window, min_periods=1).skew()
            if window > 3:
                features[f'roll_kurtosis_{window}'] = df[base_col].rolling(window=window, min_periods=1).kurt()
        features['first_derivative'] = df[self.target_column].diff()
        features['second_derivative'] = df[self.target_column].diff(2)
        for window in windows: # # todo: нужно ли тут условие if window / len_df <= 0.15: ?
            features[f'rolling_trend_{window}'] = rolling_slope(df[self.target_column].values, window)
        features['returns'] = df[self.target_column].pct_change()
        vol_window = windows[len(windows) // 2] if windows else 7
        features['volatility'] = features['returns'].rolling(window=vol_window, min_periods=1).std()

        try:
            years = pd.to_datetime(df.index).year.unique().tolist()
            for country in holiday_countries:
                country_holidays = holidays.CountryHoliday(country, years=years, expand=True, observed=True)
                features[f'{country}_is_holiday'] = pd.Series(df.index.map(lambda d: d.date() in country_holidays), index=df.index)
        except Exception:
            features['is_holiday'] = False
        logging.debug("Создание признаков завершено.")
        features = features.dropna()  # todo: сделать эксперимент: не удалять Null а заменить их на -1 или другое значение
        return features