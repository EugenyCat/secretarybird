"""
Ideal time series preparation process
Step 1: Data analysis
Check for outliers and missing values.
Determine the main properties of the series: seasonality, trend, stationarity.
Step 2: Data processing
Apply normalization (MinMaxScaler or StandardScaler) if data is scale-imbalanced.
Remove outliers, fill missing values.
Step 3: Feature enrichment
Add lags, aggregates (e.g., rolling_mean, rolling_std), and timestamps.
Step 4: Optimization for models
Perform decomposition or logarithmic transformation if data is too unstable.


worth considering adding::::

1) Lags
- need to make smart lags, for example for 3 hours ago, then 24 hours ago
- for 12 hours add 7 days ago
- for a day add a month ago
in short, make them not just plain simple lags

2) Mean/standard deviation for the last N steps. like mean_24h, std_24h

3) Periodicity and seasonality
For long time intervals (e.g., weekly or monthly) it is useful to add features that track seasonal effects:

Week period (e.g., day of the week or week of the year).
Month of the year, to capture seasonal changes.
Example:
For 1d: add a "day of the week" feature to identify day-of-week dependency.
For 1m: add "month of the year" or "week of the month".


----
1) replace statsmodels.tsa.seasonal
with - X-13ARIMA-SEATS
or Prophet

"""
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from scipy.signal import periodogram
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, acf
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class TimeSeriesPreprocessorManager:
    """
    Class for time series preprocessing. Main goal: prepare data for machine learning models,
    including filling missing values, removing outliers, scaling data, creating lag features, and decomposition.
    """

    def __init__(self, timestamp_column='Open_time', target_column='Open', additional_columns=None, scaler='minmax', lag_features=5):
        """
        Initialize the time series preprocessing manager.

        :param target_column: Main column for forecasting (e.g., 'Close').
        :param timestamp_column: Column with timestamps (e.g., 'Open_time').
        :param additional_columns: List of additional columns for processing.
        :param scaler: Type of scaling ('minmax' or 'standard').
        :param lag_features: Number of lag features to generate.
        """
        self.timestamp_column = timestamp_column
        self.target_column = target_column
        self.additional_columns = additional_columns if additional_columns else []
        self.scaler_type = scaler
        self.scaler = MinMaxScaler() if scaler == 'minmax' else StandardScaler()
        self.lag_features = lag_features
        self.original_data = None

    def analyze_series(self, data):
        """
        Analyze time series to identify missing values, outliers, and check stationarity.

        :param data: Pandas Series or array of time series data.
        :return: Dictionary with analysis results (missing values, outliers, stationarity).

        # todo: improve stationarity test by relying not only on p-value, but also on mean value
        # todo: and variance and their change over time, autocorrelation and how it decays, QQ - all should be in automatic mode
        """
        analysis = {}

        # Missing values analysis
        analysis['missing_values'] = data.isnull().sum()

        # Outliers (Z-score method)
        z_scores = zscore(data.dropna())
        analysis['outliers'] = (np.abs(z_scores) > 3).sum()

        # Stationarity (ADF test)
        adf_result = adfuller(data.dropna())
        analysis['stationary'] = adf_result[1] < 0.05  # p-value < 0.05 means stationarity

        return analysis

    def fill_missing_values(self, data, method='linear'):
        """
        Fill missing values in data using the specified interpolation method.

        :param data: Pandas Series of time series.
        :param method: Method for filling missing values ('linear', 'mean', etc.).
        :return: Series with filled missing values.

        # todo: how to make this method more flexible? as was done in remove_outliers add method parameter and organize
        # todo: several ways to fill missing values: Interpolation, Mean value, Rolling average, Prediction, Exponential smoothing
        """
        return data.interpolate(method=method).bfill().ffill()


    def _optimize_contamination(self, data, method):
        """
        Optimize the contamination parameter through data analysis.

        :param data: Pandas Series of time series.
        :param method: Method for outlier removal ('isolation_forest', 'lof').
        :return: Optimal contamination value.
        """
        """
                Optimize the contamination parameter through data analysis.

                :param data: Pandas Series of time series.
                :param method: Method for outlier removal ('isolation_forest', 'lof').
                :return: Optimal contamination value.
                """
        if not isinstance(data, pd.Series):
            data = data[self.target_column]

        # Automatic method selection
        if method == 'auto':
            method = 'isolation_forest' if len(data) > 1000 else 'lof'

        # Dynamic determination of contamination range
        iqr = data.quantile(0.75) - data.quantile(0.25)
        lower_bound = data.quantile(0.25) - 1.5 * iqr
        upper_bound = data.quantile(0.75) + 1.5 * iqr
        estimated_outliers = ((data < lower_bound) | (data > upper_bound)).sum()
        estimated_contamination = max(estimated_outliers / len(data), 0.001)

        # Generate logarithmic range of contamination
        contamination_range = np.geomspace(max(0.0001, estimated_contamination / 2),
                                           min(0.51, estimated_contamination * 2), 10)

        best_contamination = 0.001
        best_score = float('inf')

        from joblib import Parallel, delayed

        def evaluate_contamination(contamination):
            try:
                if method == 'isolation_forest':
                    processed_data = self.remove_outliers_with_isolation_forest(data.copy(), contamination)
                elif method == 'lof':
                    processed_data = self.remove_outliers_with_lof(data.copy(), contamination)
                else:
                    raise ValueError("Unsupported method for contamination optimization.")

                # Quality assessment: residual autocorrelation
                residual_diff = data - processed_data
                score = np.mean(np.abs(residual_diff))

                # Alternative metric: autocorrelation
                autocorr_score = residual_diff.autocorr()
                final_score = score * (1 - abs(autocorr_score))

                return contamination, final_score
            except Exception as e:
                print(f"Error evaluating contamination {contamination}: {e}")
                return contamination, float('inf')

        results = Parallel(n_jobs=-1)(delayed(evaluate_contamination)(cont) for cont in contamination_range)

        # Select best contamination
        for contamination, score in results:
            if score < best_score:
                best_score = score
                best_contamination = contamination

        return best_contamination


    def remove_outliers_with_isolation_forest(self, data, contamination=None):
        """
        Remove outliers using Isolation Forest with automatic contamination tuning.

        :param data: Pandas Series of time series.
        :param contamination: Expected proportion of outliers in the data. If None, it will be calculated automatically.
        :return: Series with outliers replaced by median.
        """
        if contamination is None:
            # Automatic selection of contamination through cross-validation
            contamination = self._optimize_contamination(data, method='isolation_forest')

        reshaped_data = data.values.reshape(-1, 1)
        isol_forest = IsolationForest(contamination=contamination)
        outliers = isol_forest.fit_predict(reshaped_data)
        median_value = np.median(data[~np.isin(outliers, -1)])
        data[outliers == -1] = median_value
        return data


    def remove_outliers_with_lof(self, data, contamination=None):
        """
        Remove outliers using Local Outlier Factor (LOF).

        :param data: Pandas Series of time series.
        :param contamination: Expected proportion of outliers in the data.
        :return: Series with outliers replaced by median.
        """
        if contamination is None:
            # Automatic selection of contamination through cross-validation
            contamination = self._optimize_contamination(data, method='isolation_forest')

        reshaped_data = data.values.reshape(-1, 1)
        lof = LocalOutlierFactor(contamination=contamination, n_neighbors=20)
        outliers = lof.fit_predict(reshaped_data)
        median_value = np.median(data[outliers == 1])
        data[outliers == -1] = median_value
        return data


    def decompose_and_remove_outliers(self, data, method='isolation_forest', contamination=None):
        """
        Combined method: time series decomposition and outlier removal in residuals.

        :param data: Pandas Series of time series.
        :param method: Outlier removal method ('isolation_forest', 'lof').
        :param contamination: Expected proportion of outliers. If None, it will be calculated automatically.
        :return: DataFrame with updated components (trend, seasonality, residual).
        """
        trend, seasonal, residual = self.decompose_series(data, method='STL')

        if contamination is None:
            contamination = self._optimize_contamination(residual, method)

        if method == 'isolation_forest':
            residual_cleaned = self.remove_outliers_with_isolation_forest(residual, contamination)
        elif method == 'lof':
            residual_cleaned = self.remove_outliers_with_lof(residual, contamination)
        else:
            raise ValueError("Unsupported method for outlier removal.")

        # Reassemble the time series
        cleaned_series = trend + seasonal + residual_cleaned

        return pd.DataFrame({
            f'{self.target_column}': cleaned_series
        })


    def remove_outliers(self, data, method='combined', contamination=None):
        """
        Select outlier removal method (Isolation Forest, LOF, or combined method).

        :param data: Pandas Series of time series.
        :param method: Outlier removal method ('isolation_forest', 'lof', 'combined').
        :param contamination: Expected proportion of outliers in the data. If None, it will be calculated automatically.
        :return: Series with outliers replaced by median or DataFrame (for combined method).
        """
        if method == 'combined':
            return self.decompose_and_remove_outliers(data.copy(), method='isolation_forest', contamination=contamination)
        elif method == 'isolation_forest':
            return self.remove_outliers_with_isolation_forest(data.copy(), contamination=contamination)
        elif method == 'lof':
            return self.remove_outliers_with_lof(data.copy(), contamination=contamination)
        else:
            raise ValueError("Unsupported outlier removal method.")


    def determine_period_for_decomposition(self, data, max_lags=40):
        """
        Determine the optimal seasonality period of a time series using ACF and spectral analysis.

        :param data: Pandas Series of time series.
        :param max_lags: Maximum number of lags for ACF analysis.
        :return: Optimal seasonality period.
        """
        # ACF analysis
        correlation = acf(data, nlags=max_lags, fft=True)
        significant_lags = np.where(correlation > 0.3)[0]

        # Spectral analysis
        frequencies, power = periodogram(data.dropna())
        dominant_freq = frequencies[np.argmax(power[1:]) + 1]
        period_fft = int(round(1 / dominant_freq)) if dominant_freq > 0 else 2

        # Select the smallest significant period
        if significant_lags.size > 0:
            period_acf = significant_lags[0]
            return max(min(period_acf, period_fft), 2)
        else:
            return max(period_fft, 2)

    def adapt_prophet_components(df, freq):
        """
        Adapt seasonal components for Prophet model based on data frequency.

        :param df: DataFrame with time series.
        :param freq: Data frequency ('D', 'W', 'M', etc.).
        :return: Prophet model with added components.
        """
        model = Prophet()
        if freq in ['D', 'daily']:
            model.add_seasonality(name='daily', period=1, fourier_order=5)
        if freq in ['W', 'weekly']:
            model.add_seasonality(name='weekly', period=7, fourier_order=3)
        if freq in ['M', 'monthly', 'Y', 'yearly']:
            model.add_seasonality(name='yearly', period=365.25, fourier_order=10)
        return model

    def decompose_series(self, data, method='prophet', freq=None):
        """
        Decompose time series into trend, seasonality, and residual using the specified method.

        :param data: Pandas Series of time series.
        :param method: Decomposition method ('seasonal_decompose', 'prophet', 'STL').
        :param freq: Data frequency.
        :return: Tuple of trend, seasonality, and residuals.
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex for seasonal decomposition.")

        period = self.determine_period_for_decomposition(data)

        if method == 'seasonal_decompose':
            decomposition = seasonal_decompose(data, model='additive', period=period)
            return decomposition.trend, decomposition.seasonal, decomposition.resid
        elif method == 'STL':
            stl = STL(data, period=period, seasonal=13, robust=True)
            result = stl.fit()
            return result.trend, result.seasonal, result.resid
        elif method == 'prophet':
            df = pd.DataFrame(data).reset_index()
            df.columns = ['ds', 'y']
            model = self.adapt_prophet_components(freq)
            model.fit(df)
            future = model.make_future_dataframe(periods=0)
            forecast = model.predict(future)

            trend = forecast.get('trend', 0)
            seasonal_components = ['yearly', 'weekly', 'daily']
            seasonal = sum([forecast.get(component, 0) for component in seasonal_components])

            resid = df['y'] - trend - seasonal
            return trend, seasonal, resid
        else:
            raise ValueError("Invalid decomposition method specified.")

    def scale_data(self, data):
        """
        Scale data using the specified scaler.

        :param data: Pandas Series of time series.
        :return: Scaled data and trained scaler.
        """
        reshaped_data = data.values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(reshaped_data)
        return scaled_data.flatten()

    def inverse_scale(self, scaled_data):
        """
        Inverse transform scaled data back to original scale.

        :param scaled_data: Scaled data.
        :return: Data in original scale.
        """
        reshaped_data = scaled_data.reshape(-1, 1)
        return self.scaler.inverse_transform(reshaped_data).flatten()

    def add_lag_features(self, data):
        """
        Add lag features to data.

        :param data: Pandas Series of time series.
        :return: DataFrame with original data and lag features.
        """
        df = pd.DataFrame(data, columns=[f'{self.target_column}_scaled'])

        for lag in range(1, self.lag_features + 1):
            df[f'lag_{lag}'] = df[f'{self.target_column}_scaled'].shift(lag)

        return df.dropna()

    def process(self, df):
        """
        Main data preprocessing method.

        :param df: DataFrame with time series containing the following columns:
            - Open_time and Close_time: Interval start and end time.
            - Open, High, Low, Close: Open, high, low, and close prices.
            - Volume: Trading volume in base currency.
            - Quote_asset_volume: Trading volume in quote currency.
            - Number_of_trades: Number of trades per interval.
            - Interval: Data interval (e.g., 1h, 12h, 1d).
            - Source: Data source (e.g., 'binance_api').
        :return: Tuple (X, y) with features and target values for model training.
        """
        try:
            self.original_data = df.copy()

            if self.timestamp_column not in df.columns:
                raise ValueError(f"Timestamp column '{self.timestamp_column}' not found in the DataFrame.")

            if not isinstance(df[self.timestamp_column], pd.DatetimeIndex):
                df[self.timestamp_column] = pd.to_datetime(df[self.timestamp_column], errors='coerce')

            columns_to_process = [self.target_column] + self.additional_columns
            processed_data = df[columns_to_process].copy()
            timestamps = df[self.timestamp_column]

            X = pd.DataFrame(index=timestamps.index)
            X[self.timestamp_column] = timestamps

            for col in columns_to_process:
                data = processed_data[col]
                analysis = self.analyze_series(data)
                data = self.fill_missing_values(data)
                data = pd.DataFrame({col: data})
                data.set_index(timestamps, inplace=True)

                data = self.remove_outliers(data)

                trend, seasonal, residual = self.decompose_series(data[col])

                processed_data[f'{col}_residual'] = residual.reset_index(drop=True).fillna(0)
                processed_data[f'{col}_scaled'] = self.scale_data(processed_data[f'{col}_residual'])

                X[f'{col}_scaled'] = processed_data[f'{col}_scaled']

            lagged_data = self.add_lag_features(processed_data[f'{self.target_column}_scaled'])
            X = X.join(lagged_data, how='inner', lsuffix='_original', rsuffix='_lag')

            y = df[self.target_column]

            return X, y
        except Exception as e:
            raise Exception(f'[ml_manager/TimeSeriesPreprocessorManager] error: {e}')