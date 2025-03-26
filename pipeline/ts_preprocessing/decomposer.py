
# -----------------------------------------------------------------------------
# Класс для детрендинга и декомпозиции
# TrendDecomposer и стратегии декомпозиции (STL, SSA, Fourier и пр.)
# -----------------------------------------------------------------------------

from pipeline.ts_preprocessing.periodicity import PeriodicityDetector
import pandas as pd
import numpy as np
import logging
from statsmodels.tsa.stattools import acf, adfuller
import statsmodels.api as sm
from numpy.lib.stride_tricks import as_strided
from numba import njit


@njit
def hankel_diagonal_averaging(X):
    window_length, K = X.shape
    N = window_length + K - 1
    reconstructed = np.zeros(N)
    counts = np.zeros(N)
    for i in range(window_length):
        for j in range(K):
            reconstructed[i + j] += X[i, j]
            counts[i + j] += 1.0
    for i in range(N):
        reconstructed[i] /= counts[i]
    return reconstructed

class TrendDecomposer:

    def __init__(self, interval):
        self.interval = interval

    def detrend_series(self, data: pd.Series, method='linear'):
        logging.debug("Начало детрендинга методом: %s", method)
        if method == 'linear':
            x = np.arange(len(data))
            coeffs = np.polyfit(x, data.values, 1)
            trend = np.polyval(coeffs, x)
            detrended = data - trend
            logging.debug("Линейный детрендинг завершён.")
            return detrended, pd.Series(trend, index=data.index)
        elif method == 'difference':
            detrended = data.diff().dropna()
            logging.debug("Детрендинг через разности завершён.")
            return detrended, data.iloc[0]
        else:
            raise ValueError("Unsupported detrending method. Choose 'linear' or 'difference'.")

    def inverse_detrend(self, detrended_data, trend_component, method='linear'):
        logging.debug("Выполняется обратное детрендинг методом: %s", method)
        if method == 'linear':
            return detrended_data + trend_component
        elif method == 'difference':
            return detrended_data.cumsum() + trend_component
        else:
            raise ValueError("Unsupported inverse detrending method. Choose 'linear' or 'difference'.")

    def decompose_series(self, data: pd.Series, method='auto'):
        logging.debug("Начало декомпозиции временного ряда (метод: %s).", method)
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex for seasonal decomposition.")
        adf_pvalue = adfuller(data.dropna())[1]
        model_type = 'additive' if adf_pvalue < 0.05 else 'multiplicative'
        # Определяем период – здесь используется упрощённая версия
        period, period_candidates = self._determine_period(data)
        method_to_use = self._determine_method_auto(data, period, period_candidates) if method == 'auto' else method
        logging.debug("Выбран метод декомпозиции: %s", method_to_use)
        decomposition_methods = {
            'STL': self._decompose_stl,
            'TBATS': self._decompose_tbats,
            'SSA': self._decompose_ssa,
            'fourier': self._decompose_fourier,
            'prophet': self._decompose_prophet
        }
        if method_to_use in decomposition_methods:
            return decomposition_methods[method_to_use](
                data=data,
                period=period,
                period_candidates=period_candidates,
                model_type=model_type
            )
        else:
            raise ValueError(f"Unsupported decomposition method: {method_to_use}")

    def _determine_method_auto(self, data, period, candidates):
        if period < 2:
            return 'SSA'
        methods = ['STL', 'SSA', 'fourier']
        quality = {}
        for m in methods:
            try:
                result = getattr(self, f'_decompose_{m.lower()}')(data, period, candidates)
                quality[m] = np.nanmean(np.abs(result[2].dropna()))
            except Exception as e:
                quality[m] = np.inf
        chosen = min(quality, key=quality.get)
        logging.debug("Метрики качества декомпозиции: %s, выбран метод: %s", quality, chosen)
        return chosen

    def _determine_period(self, data):
        # Используем PeriodicityDetector, ACF и FFT
        # (Предполагается, что PeriodicityDetector определён где-то в окружении)
        detector = PeriodicityDetector(min_period=2, max_period=None, cv=10)
        detector.fit(data)
        period_detector = detector.predict(data)
        period_candidates = []
        if period_detector > 1:
            logging.debug(f"Применение PeriodicityDetector: {period_detector}")
            period_candidates.append(period_detector)
        acf_vals = acf(data.dropna(), nlags=min(100, len(data) // 2))
        for lag, val in enumerate(acf_vals[1:], start=1):
            if val > 0.3:
                logging.debug(f"[LOG] Применение ACF: выбран lag={lag}")
                period_candidates.append(lag)
                break
        fft_data = data.dropna().values
        fft_vals = np.abs(np.fft.rfft(fft_data))
        freqs = np.fft.rfftfreq(len(fft_data), d=1)
        if len(fft_vals) > 1:
            index_candidate = np.argmax(fft_vals[1:]) + 1
            if index_candidate >= len(freqs):
                index_candidate = len(freqs) - 1
            dominant_freq = freqs[index_candidate]
            default_max_period = {
                '1h': 168,  # 24*7 – неделя для часовых данных
                '12h': 60,  # примерно 30 дней
                '1d': 182,  # примерно полгода для ежедневных данных
                '3d': 60,  # примерно 180 дней/3 ≈ 60 наблюдений для 3-дневного интервала
                '1w': 52,  # 52 недели (год) для недельных данных
                '1M': 12  # 12 месяцев для месячных данных
            }
            max_allowed_period = default_max_period.get(self.interval, len(data) / 2)
            if dominant_freq > 0:
                period_fft = int(round(1 / dominant_freq))
                if period_fft < max_allowed_period:
                    # print(f"[LOG] Применение FFT: период={period_fft}")
                    period_candidates.append(period_fft)
                else:
                    logging.debug(f"[LOG] Игнорируем кандидат FFT: {period_fft} (превышает порог)")
        if period_candidates:
            period = int(np.round(np.median(period_candidates)))
        else:
            period = 0
        if period < 2:
            logging.debug("[LOG] Применение Фолбэка для периода")
            default_periods = {'1h': 24, '12h': 14, '1d': 7, '3d': 7, '1w': 52, '1M': 12}
            period = default_periods.get(self.interval, 1)
        logging.debug(f"[LOG] Определённый период: {period}, кандидаты: {period_candidates}")
        return period, period_candidates


    def _decompose_stl(self, data, period, **kwargs):
        logging.debug("Начало STL декомпозиции.")
        from statsmodels.tsa.seasonal import STL
        stl = STL(data, period=period, robust=True)
        result = stl.fit()
        logging.debug("STL декомпозиция завершена.")
        return result.trend, result.seasonal, result.resid

    def _decompose_tbats(self, data, period, candidates, **kwargs):
        logging.debug("Начало TBATS декомпозиции.")
        from tbats import TBATS
        seasonal_periods = candidates if len(candidates) > 1 else [period]
        seasonal_periods = [24, 168] if seasonal_periods == [period] and period == 24 else seasonal_periods
        estimator = TBATS(seasonal_periods=seasonal_periods)
        tbats_model = estimator.fit(data)
        fitted = tbats_model.y_hat
        try:
            seasonal_array = np.sum(tbats_model.seasonal_components_, axis=1)
        except AttributeError:
            seasonal_array = np.zeros_like(fitted)
        trend_array = fitted - seasonal_array
        resid_array = data.values.flatten() - fitted
        logging.debug("TBATS декомпозиция завершена.")
        trend = pd.Series(trend_array, index=data.index)
        seasonal = pd.Series(seasonal_array, index=data.index)
        resid = pd.Series(resid_array, index=data.index)
        return trend, seasonal, resid

    def _decompose_ssa(self, data, period, **kwargs):
        logging.debug("Начало SSA декомпозиции.")
        return self.ssa_decompose(data, window_length=period, variance_threshold=0.9)

    def _decompose_fourier(self, data, period, **kwargs):
        logging.debug("Начало Fourier декомпозиции.")
        result = self.fourier_decompose(data, period=period)
        logging.debug("Fourier декомпозиция завершена")
        return result

    def _decompose_prophet(self, data, **kwargs):
        logging.debug("Начало Prophet декомпозиции.")
        from prophet import Prophet
        df = pd.DataFrame(data)
        df.reset_index(inplace=True)
        df.columns = ['ds', 'y']
        data_length = len(df)
        seasonality_config = self.get_seasonality_config(self.interval, data_length)
        components = [comp for comp, flag in seasonality_config.items() if flag]
        model_prophet = Prophet(
            daily_seasonality=seasonality_config.get('daily', False),
            weekly_seasonality=seasonality_config.get('weekly', False),
            yearly_seasonality=seasonality_config.get('yearly', False),
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            seasonality_mode='multiplicative'
        )
        model_prophet.fit(df)
        forecast = model_prophet.predict(df)
        trend_series = pd.Series(forecast['trend'].values, index=data.index)
        seasonal_series = pd.Series(forecast[components].sum(axis=1).values, index=data.index)
        resid_series = pd.Series((df['y'] - forecast['trend'] - forecast[components].sum(axis=1)).values, index=data.index)
        logging.debug("Prophet декомпозиция завершена.")
        return trend_series, seasonal_series, resid_series

    def get_seasonality_config(self, interval, data_length):
        configs = {
            '1h': {'daily': True, 'weekly': False, 'yearly': False},
            '12h': {'daily': True, 'weekly': True, 'yearly': False},
            '1d': {'daily': False, 'weekly': True, 'yearly': True},
            '3d': {'daily': False, 'weekly': True, 'yearly': True},
            '1w': {'daily': False, 'weekly': False, 'yearly': True},
            '1M': {'daily': False, 'weekly': False, 'yearly': True}
        }
        config = configs.get(interval, {'daily': False, 'weekly': False, 'yearly': False})
        if data_length < 365:
            config['yearly'] = False
        return config

    def fourier_decompose(self, data, period):
        logging.debug("Детализированное Fourier декомпозиция начинается.")
        data_series = pd.Series(data.values.flatten(), index=data.index)
        df = pd.DataFrame({'y': data_series.values}, index=data_series.index)
        t = np.arange(len(df))
        max_k = 5
        const = np.ones_like(t)
        harmonics = np.arange(1, max_k+1)
        sin_full = np.sin(2 * np.pi * np.outer(t, harmonics) / period)
        cos_full = np.cos(2 * np.pi * np.outer(t, harmonics) / period)
        best_aic = np.inf
        best_model = None
        best_X = None
        for k in range(1, max_k+1):
            X = np.column_stack((const, t, sin_full[:, :k], cos_full[:, :k]))
            X = sm.add_constant(X, has_constant='add')
            model_fit = sm.OLS(df['y'], X).fit()
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_model = model_fit
                best_X = X.copy()
        params = best_model.params
        trend_values = best_X[:, 1] * params[1] + best_X[:, 0]
        seasonal_values = best_X[:, 2:] @ params[2:]
        logging.debug("Fourier декомпозиция завершена.")
        return (pd.Series(trend_values, index=df.index),
                pd.Series(seasonal_values, index=df.index),
                data_series - pd.Series(trend_values, index=df.index) - pd.Series(seasonal_values, index=df.index))

    def ssa_decompose(self, data, window_length=None, variance_threshold=0.9):
        logging.debug("Начало SSA декомпозиции.")
        series = data.values.flatten() if isinstance(data, pd.Series) else np.array(data).flatten()
        N = series.shape[0]
        if window_length is None:
            period, _ = self._determine_period(data)
            window_length = period if period > 1 else N // 2
        K = N - window_length + 1
        X = as_strided(series, shape=(window_length, K), strides=(series.strides[0], series.strides[0])).copy()
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        total_variance = np.sum(s**2)
        explained_variance = np.cumsum(s**2) / total_variance
        r = np.searchsorted(explained_variance, variance_threshold) + 1
        X_reconstructed = np.zeros_like(X)
        for i in range(r):
            X_reconstructed += s[i] * np.outer(U[:, i], Vt[i, :])
        reconstructed = hankel_diagonal_averaging(X_reconstructed)
        if r >= 2:
            X_trend = s[0] * np.outer(U[:, 0], Vt[0, :]) + s[1] * np.outer(U[:, 1], Vt[1, :])
            trend_reconstructed = hankel_diagonal_averaging(X_trend)
            seasonal_reconstructed = reconstructed - trend_reconstructed
        else:
            trend_reconstructed = reconstructed
            seasonal_reconstructed = np.zeros_like(reconstructed)
        residual = series - reconstructed
        logging.debug("SSA декомпозиция завершена.")
        if hasattr(data, 'index'):
            trend_reconstructed = pd.Series(trend_reconstructed, index=data.index)
            seasonal_reconstructed = pd.Series(seasonal_reconstructed, index=data.index)
            residual = pd.Series(residual, index=data.index)
        return trend_reconstructed, seasonal_reconstructed, residual