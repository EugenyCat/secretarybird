
# -----------------------------------------------------------------------------
# Фасад для предобработки временных рядов
# Фасад TimeSeriesPreprocessor, объединяющий все модули
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import logging

from pipeline.ts_preprocessing.ts_analyzer import TimeSeriesAnalyzer
from pipeline.ts_preprocessing.outlier_remover import OutlierRemover
from pipeline.ts_preprocessing.feature_engineer import FeatureEngineer
from pipeline.ts_preprocessing.periodicity import PeriodicityEstimator, PeriodicityDetector

class TimeSeriesPreprocessor:
    """
    Фасад для предобработки временных рядов. Объединяет анализ, детрендинг,
    декомпозицию, удаление выбросов, масштабирование, создание признаков и оценку периодичности.
    """
    def __init__(self, interval, timestamp_column='Open_time', target_column='Open', additional_columns=None, scaler='robust', lag_features=5):
        self.interval = interval
        self.timestamp_column = timestamp_column
        self.target_column = target_column
        self.additional_columns = additional_columns if additional_columns else []
        self.scaler_type = scaler
        self.lag_features = lag_features
        self.analyzer = TimeSeriesAnalyzer()
        self.outlier_remover = OutlierRemover()
        self.feature_engineer = FeatureEngineer(scaler_type=scaler, lag_features=lag_features, interval=interval, target_column=target_column)
        self.periodicity_estimator = PeriodicityEstimator()
        self.periodicity_detector = PeriodicityDetector(min_period=2, max_period=None, cv=10)
        self.original_data = None

    def process(self, df: pd.DataFrame):
        try:
            logging.info("Начало пайплайна предобработки временного ряда.")
            df = df.copy()
            if self.timestamp_column not in df.columns:
                raise ValueError(f"Timestamp column '{self.timestamp_column}' missing.")
            if not np.issubdtype(df[self.timestamp_column].dtype, np.datetime64):
                df[self.timestamp_column] = pd.to_datetime(df[self.timestamp_column], errors='coerce')
            self.original_data = df.copy()
            df.set_index(self.timestamp_column, inplace=True)
            for col in [self.target_column] + self.additional_columns:
                df[col] = df[col].interpolate(method='linear').bfill().ffill()
            #self.analyzer.analyze_series(df[self.target_column]) # этот метод просто делает анализ, но не изменяет и не дополняет входные данные
            df[f'{self.target_column}_removed_outliers'] = self.outlier_remover.remove_outliers(df[self.target_column], self.interval)
            scaled_series = self.feature_engineer.scale_data(df[f'{self.target_column}_removed_outliers'])
            df[f'{self.target_column}_scaled'] = scaled_series
            df = self.feature_engineer.add_features(df)
            df.sort_index(inplace=True)
            logging.debug("Пайплайн завершён. Количество наблюдений: %d", len(df))
            y = df[f'{self.target_column}_scaled']
            X = df.drop(columns=[f'{self.target_column}_scaled'])
            return X, y
        except Exception as e:
            raise Exception(f'[ml_manager/TimeSeriesPreprocessorManager] error: {e}')