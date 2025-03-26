
# -----------------------------------------------------------------------------
# Классы для оценки периодичности
# PeriodicityEstimator и PeriodicityDetector
# -----------------------------------------------------------------------------

import numpy as np
import logging
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import STL
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from scipy.signal import find_peaks
from numba import njit


@njit
def select_period(acf_values, min_period, max_period_loc, period_threshold): #+
    n = acf_values.shape[0]
    best_period = 0
    best_acf = 0.0
    found_strong = False
    # Перебираем индексы от min_period до max_period_loc (включительно)
    for i in range(min_period, min(max_period_loc+1, n)):
        # Простая проверка "вершины": если значение больше предыдущего и не меньше следующего
        if i > 0 and i < n - 1:
            if acf_values[i] > acf_values[i - 1] and acf_values[i] >= acf_values[i + 1]:
                if acf_values[i] >= period_threshold and not found_strong:
                    best_period = i
                    found_strong = True
                elif not found_strong:
                    if acf_values[i] > best_acf:
                        best_acf = acf_values[i]
                        best_period = i
    return best_period

class PeriodicityEstimator(BaseEstimator): # убран TransformerMixin т.к transform не нужен
    def __init__(self, max_nlags_ratio=0.5, peak_prominence_threshold=0.1,
                 period_threshold=0.8, min_period=2, max_period=None):
        self.max_nlags_ratio = max_nlags_ratio
        self.peak_prominence_threshold = peak_prominence_threshold
        self.period_threshold = period_threshold
        self.min_period = min_period
        self.max_period = max_period
        self.period_ = None # для хранения найденного периода

    def fit(self, X, y=None):
        # метод fit ничего не делает, кроме как сохраняет переданные параметры
        # обучение происходит в GridSearchCV
        return self

    def predict(self, X):
        logging.debug("Начало оценки периода.")
        period = self._find_period(X)
        self.period_ = period
        logging.debug("Оценённый период: %d", period)
        return period

    def _find_period(self, data):
        n = len(data)
        max_nlags = int(self.max_nlags_ratio * n)
        max_period_loc = n // 2 if self.max_period is None else self.max_period
        acf_values = acf(data, nlags=max_nlags, fft=True)

        # option 1: быстрее определяет period но без использования peak_prominence_threshold
        #period = select_period(acf_values, self.min_period, max_period_loc, self.period_threshold)

        # option 2: дольше но с peak_prominence_threshold
        peaks, _ = find_peaks(acf_values, prominence=self.peak_prominence_threshold)

        valid_peaks = peaks[(peaks >= self.min_period) & (peaks <= max_period_loc)]

        if not valid_peaks.size:
            period = 0
        else:
            acf_peak_values = acf_values[valid_peaks]
            strong_peaks = np.where(acf_peak_values >= self.period_threshold)[0]
            if strong_peaks.size > 0:
                period = valid_peaks[strong_peaks[0]]
            else:
                period = valid_peaks[np.argmax(acf_peak_values)]

        # Возврат period
        return period


class PeriodicityDetector(BaseEstimator):
    def __init__(self, min_period=2, max_period=None, cv=5, downsample_factor=10):
        self.min_period = min_period
        self.max_period = max_period
        self.cv = cv
        self.downsample_factor = downsample_factor
        self.best_estimator_ = None

    def fit(self, X, y=None):
        logging.debug("Начало GridSearch для PeriodicityDetector.")
        param_grid = {
            'max_nlags_ratio': [0.1, 0.25, 0.5],
            'peak_prominence_threshold': [0.05, 0.1, 0.2],
            'period_threshold': [0.5, 0.6, 0.7, 0.8],
            'min_period': [self.min_period],
            'max_period': [self.max_period]
        }
        estimator = PeriodicityEstimator()
        grid_search = GridSearchCV(estimator, param_grid, scoring=self._score_period, cv=self.cv, n_jobs=-1)
        # Передаём данные как одномерный массив
        grid_search.fit(X.to_numpy().ravel().reshape(-1, 1), y=[0] * len(X))
        self.best_estimator_ = grid_search.best_estimator_
        logging.debug("GridSearch завершён. Лучший период: %s", self.best_estimator_.period_)
        return self

    def _score_period(self, y, period_, X=None, **kwargs):
        period = period_[0]
        if period == 0:
            return -np.inf
        try:
            # Downsampling для ускорения STL
            X_ds = X[::self.downsample_factor]
            stl = STL(X_ds, period=period)
            res = stl.fit()
            return -res.resid.var()
        except Exception:
            return -np.inf

    def predict(self, X):
        logging.debug("Предсказание периода по лучшему оценщику.")
        return self.best_estimator_.predict(X)