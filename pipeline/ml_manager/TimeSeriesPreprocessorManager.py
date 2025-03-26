"""
Идеальный процесс подготовки временного ряда
Шаг 1: Анализ данных
Проверить наличие выбросов и пропусков.
Определить основные свойства ряда: сезонность, тренд, стационарность.
Шаг 2: Обработка данных
Применить нормализацию (MinMaxScaler или StandardScaler), если данные масштабно несбалансированы.
Удалить выбросы, заполнить пропуски.
Шаг 3: Обогащение признаков
Добавить лаги, агрегаты (например, rolling_mean, rolling_std), и временные метки.
Шаг 4: Оптимизация для моделей
Провести декомпозицию или логарифмическую трансформацию, если данные слишком нестабильны.


стоит рассмотреть добалвение::::

1) Лаги
- надо сделать умные лаги например дл 3 часов назад, потом 24 часа назад
- для 12 часов добавить 7 дней назад
- для дня добавить месяц назад
короче сделать их не тупо прям просто лаги

2) Среднее/среднеквадратичное отклонение за последние N шагов. типа mean_24h, std_24h

3) Периодичность и сезонность
Для длинных временных интервалов (например, недельный или месячный) полезно добавлять фичи, которые отслеживают сезонные эффекты:

Период недели (например, день недели или неделя года).
Месяц года, чтобы уловить сезонные изменения.
Пример:
Для 1д: добавьте фичу "день недели", чтобы выявить зависимость от дня недели.
Для 1м: добавьте "месяц года" или "неделя месяца".


----
1) заменить statsmodels.tsa.seasonal
на - X-13ARIMA-SEATS
или Prophet

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
    Класс для предобработки временных рядов. Основная цель: подготовить данные для моделей машинного обучения,
    включая заполнение пропусков, удаление выбросов, масштабирование данных, создание лаговых признаков и декомпозицию.
    """

    def __init__(self, timestamp_column='Open_time', target_column='Open', additional_columns=None, scaler='minmax', lag_features=5):
        """
        Инициализация менеджера предобработки временных рядов.

        :param target_column: Основная колонка для прогнозирования (например, 'Close').
        :param timestamp_column: Колонка с временными метками (например, 'Open_time').
        :param additional_columns: Список дополнительных колонок для обработки.
        :param scaler: Тип масштабирования ('minmax' или 'standard').
        :param lag_features: Количество лаговых признаков для генерации.
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
        Анализ временного ряда для выявления пропусков, выбросов и проверки стационарности.

        :param data: Pandas Series или массив данных временного ряда.
        :return: Словарь с результатами анализа (пропуски, выбросы, стационарность).

        # todo: улучшить тест на стационарность опираясь не только на p-value, но также и на среднее значение
        # todo: и дисперсию и их изменение во времени , автокорреляцию и то как он падает, QQ - все должно быть в автоматическом режиме
        """
        analysis = {}

        # Анализ пропусков
        analysis['missing_values'] = data.isnull().sum()

        # Выбросы (метод Z-оценки)
        z_scores = zscore(data.dropna())
        analysis['outliers'] = (np.abs(z_scores) > 3).sum()

        # Стационарность (тест ADF)
        adf_result = adfuller(data.dropna())
        analysis['stationary'] = adf_result[1] < 0.05  # p-value < 0.05 означает стационарность

        return analysis

    def fill_missing_values(self, data, method='linear'):
        """
        Заполнение пропусков в данных указанным методом интерполяции.

        :param data: Pandas Series временного ряда.
        :param method: Метод заполнения пропусков ('linear', 'mean' и т.д.).
        :return: Series с заполненными пропусками.

        # todo: как сделать этот метод более гибким? как это было сделано в remove_outliers добавить параметр method и организовать
        # todo: несколько способов заполнения пропусков: Интерполяция, Среднее значение, Скользящее среднее, Предсказание, Экспоненциальное сглаживание
        """
        return data.interpolate(method=method).bfill().ffill()


    def _optimize_contamination(self, data, method):
        """
        Оптимизация параметра contamination через разделение на обучающие и тестовые данные.

        :param data: Pandas Series временного ряда.
        :param method: Метод для удаления выбросов ('isolation_forest', 'lof').
        :return: Оптимальное значение contamination.
        """
        """
                Оптимизация параметра contamination через анализ данных.

                :param data: Pandas Series временного ряда.
                :param method: Метод для удаления выбросов ('isolation_forest', 'lof').
                :return: Оптимальное значение contamination.
                """
        if not isinstance(data, pd.Series):
            data = data[self.target_column]

        # Автоматический выбор метода
        if method == 'auto':
            method = 'isolation_forest' if len(data) > 1000 else 'lof'

        # Динамическое определение диапазона contamination
        iqr = data.quantile(0.75) - data.quantile(0.25)
        lower_bound = data.quantile(0.25) - 1.5 * iqr
        upper_bound = data.quantile(0.75) + 1.5 * iqr
        estimated_outliers = ((data < lower_bound) | (data > upper_bound)).sum()
        estimated_contamination = max(estimated_outliers / len(data), 0.001)

        # Генерация логарифмического диапазона contamination
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

                # Оценка качества: автокорреляция остатков
                residual_diff = data - processed_data
                score = np.mean(np.abs(residual_diff))

                # Альтернативная метрика: автокорреляция
                autocorr_score = residual_diff.autocorr()
                final_score = score * (1 - abs(autocorr_score))

                return contamination, final_score
            except Exception as e:
                print(f"Error evaluating contamination {contamination}: {e}")
                return contamination, float('inf')

        results = Parallel(n_jobs=-1)(delayed(evaluate_contamination)(cont) for cont in contamination_range)

        # Выбор наилучшего contamination
        for contamination, score in results:
            if score < best_score:
                best_score = score
                best_contamination = contamination

        return best_contamination


    def remove_outliers_with_isolation_forest(self, data, contamination=None):
        """
        Удаление выбросов с использованием Isolation Forest с автоматической настройкой contamination.

        :param data: Pandas Series временного ряда.
        :param contamination: Ожидаемая доля выбросов в данных. Если None, она будет рассчитана автоматически.
        :return: Series с выбросами, заменёнными на медиану.
        """
        if contamination is None:
            # Автоматический выбор contamination через кросс-валидацию
            contamination = self._optimize_contamination(data, method='isolation_forest')

        reshaped_data = data.values.reshape(-1, 1)
        isol_forest = IsolationForest(contamination=contamination)
        outliers = isol_forest.fit_predict(reshaped_data)
        median_value = np.median(data[~np.isin(outliers, -1)])
        data[outliers == -1] = median_value
        return data


    def remove_outliers_with_lof(self, data, contamination=None):
        """
        Удаление выбросов с использованием Local Outlier Factor (LOF).

        :param data: Pandas Series временного ряда.
        :param contamination: Ожидаемая доля выбросов в данных.
        :return: Series с выбросами, заменёнными на медиану.
        """
        if contamination is None:
            # Автоматический выбор contamination через кросс-валидацию
            contamination = self._optimize_contamination(data, method='isolation_forest')

        reshaped_data = data.values.reshape(-1, 1)
        lof = LocalOutlierFactor(contamination=contamination, n_neighbors=20)
        outliers = lof.fit_predict(reshaped_data)
        median_value = np.median(data[outliers == 1])
        data[outliers == -1] = median_value
        return data


    def decompose_and_remove_outliers(self, data, method='isolation_forest', contamination=None):
        """
        Комбинированный метод: декомпозиция временного ряда и удаление выбросов в остатках.

        :param data: Pandas Series временного ряда.
        :param method: Метод удаления выбросов ('isolation_forest', 'lof').
        :param contamination: Ожидаемая доля выбросов. Если None, она будет рассчитана автоматически.
        :return: DataFrame с обновлёнными компонентами (тренд, сезонность, остаток).
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

        # Сбор временного ряда обратно
        cleaned_series = trend + seasonal + residual_cleaned

        return pd.DataFrame({
            f'{self.target_column}': cleaned_series
        })


    def remove_outliers(self, data, method='combined', contamination=None):
        """
        Выбор метода удаления выбросов (Isolation Forest, LOF или комбинированный метод).

        :param data: Pandas Series временного ряда.
        :param method: Метод удаления выбросов ('isolation_forest', 'lof', 'combined').
        :param contamination: Ожидаемая доля выбросов в данных. Если None, она будет рассчитана автоматически.
        :return: Series с выбросами, заменёнными на медиану или DataFrame (для комбинированного метода).
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
        Определение оптимального периода сезонности временного ряда с использованием ACF и спектрального анализа.

        :param data: Pandas Series временного ряда.
        :param max_lags: Максимальное количество лагов для анализа ACF.
        :return: Оптимальный период сезонности.
        """
        # ACF анализ
        correlation = acf(data, nlags=max_lags, fft=True)
        significant_lags = np.where(correlation > 0.3)[0]

        # Спектральный анализ
        frequencies, power = periodogram(data.dropna())
        dominant_freq = frequencies[np.argmax(power[1:]) + 1]
        period_fft = int(round(1 / dominant_freq)) if dominant_freq > 0 else 2

        # Выбор наименьшего значимого периода
        if significant_lags.size > 0:
            period_acf = significant_lags[0]
            return max(min(period_acf, period_fft), 2)
        else:
            return max(period_fft, 2)

    def adapt_prophet_components(df, freq):
        """
        Адаптация сезонных компонентов для модели Prophet на основе частоты данных.

        :param df: DataFrame с временным рядом.
        :param freq: Частота данных ('D', 'W', 'M' и т.д.).
        :return: Модель Prophet с добавленными компонентами.
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
        Декомпозиция временного ряда на тренд, сезонность и остаток указанным методом.

        :param data: Pandas Series временного ряда.
        :param method: Метод декомпозиции ('seasonal_decompose', 'prophet', 'STL').
        :param freq: Частота данных.
        :return: Кортеж из тренда, сезонности и остатков.
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
        Масштабирование данных с использованием заданного скейлера.

        :param data: Pandas Series временного ряда.
        :return: Масштабированные данные и обученный скейлер.
        """
        reshaped_data = data.values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(reshaped_data)
        return scaled_data.flatten()

    def inverse_scale(self, scaled_data):
        """
        Обратное преобразование масштабированных данных к исходной шкале.

        :param scaled_data: Масштабированные данные.
        :return: Данные в исходной шкале.
        """
        reshaped_data = scaled_data.reshape(-1, 1)
        return self.scaler.inverse_transform(reshaped_data).flatten()

    def add_lag_features(self, data):
        """
        Добавление лаговых признаков к данным.

        :param data: Pandas Series временного ряда.
        :return: DataFrame с исходными данными и лаговыми признаками.
        """
        df = pd.DataFrame(data, columns=[f'{self.target_column}_scaled'])

        for lag in range(1, self.lag_features + 1):
            df[f'lag_{lag}'] = df[f'{self.target_column}_scaled'].shift(lag)

        return df.dropna()

    def process(self, df):
        """
        Основной метод предобработки данных.

        :param df: DataFrame с временными рядами, содержащий следующие столбцы:
            - Open_time и Close_time: Время начала и окончания интервала.
            - Open, High, Low, Close: Цены открытия, максимальная, минимальная и закрытия.
            - Volume: Объём торгов в базовой валюте.
            - Quote_asset_volume: Объём торгов в котируемой валюте.
            - Number_of_trades: Количество торгов за интервал.
            - Interval: Интервал данных (например, 1h, 12h, 1d).
            - Source: Источник данных (например, 'binance_api').
        :return: Tuple (X, y) с признаками и целевыми значениями для обучения моделей.
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

