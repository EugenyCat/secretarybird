# -----------------------------------------------------------------------------
# Класс для удаления выбросов
# OutlierRemover и связанные функции
# -----------------------------------------------------------------------------

from pipeline.ts_preprocessing.decomposer import TrendDecomposer
import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from joblib import Parallel, delayed


class OutlierRemover:
    def _optimize_contamination(self, data: pd.Series, target_column=None, method='isolation_forest'):
        logging.debug("Оптимизация contamination для метода: %s", method)
        if not isinstance(data, pd.Series):
            raise ValueError("Data should be a pandas Series.")
        iqr = data.quantile(0.75) - data.quantile(0.25)
        lower_bound = data.quantile(0.25) - 1.5 * iqr
        upper_bound = data.quantile(0.75) + 1.5 * iqr
        estimated_outliers = ((data < lower_bound) | (data > upper_bound)).sum()
        estimated_contamination = max(estimated_outliers / len(data), 0.001)
        contamination_range = np.geomspace(max(0.0001, estimated_contamination / 2),
                                           min(0.5, estimated_contamination * 2), 10)
        best_contamination = 0.001
        best_score = float('inf')

        def evaluate_contamination(contamination):
            try:
                if method == 'isolation_forest':
                    processed_data = self.remove_outliers_with_isolation_forest(data.copy(), contamination)
                elif method == 'lof':
                    processed_data = self.remove_outliers_with_lof(data.copy(), contamination)
                else:
                    raise ValueError("Unsupported method for contamination optimization.")
                residual_diff = data - processed_data
                score = np.mean(np.abs(residual_diff))
                autocorr_score = residual_diff.autocorr()
                final_score = score * (1 - abs(autocorr_score))
                return contamination, final_score
            except Exception as e:
                logging.error("Ошибка при оценке contamination %s: %s", contamination, e)
                return contamination, float('inf')

        results = Parallel(n_jobs=-1)(delayed(evaluate_contamination)(cont) for cont in contamination_range)
        for contamination, score in results:
            if score < best_score:
                best_score = score
                best_contamination = contamination
        logging.debug("Оптимизированное значение contamination: %f", best_contamination)
        return best_contamination

    def remove_outliers_with_isolation_forest(self, data: pd.Series, contamination=None):
        logging.debug("Удаление выбросов методом IsolationForest.")
        if contamination is None:
            contamination = self._optimize_contamination(data, method='isolation_forest')
        reshaped_data = data.values.reshape(-1, 1)
        isol_forest = IsolationForest(contamination=contamination)
        outliers = isol_forest.fit_predict(reshaped_data)
        median_value = np.median(data[~np.isin(outliers, -1)])
        data[outliers == -1] = median_value
        logging.debug("Выбросы удалены (IsolationForest).")
        return data

    def remove_outliers_with_lof(self, data: pd.Series, contamination=None, n_neighbors=None):
        logging.debug("Удаление выбросов методом LOF.")
        if contamination is None:
            contamination = self._optimize_contamination(data, method='lof')
        if n_neighbors is None:
            n_neighbors = min(20, max(5, int(len(data) * 0.05)))
        reshaped_data = data.values.reshape(-1, 1)
        lof = LocalOutlierFactor(contamination=contamination, n_neighbors=n_neighbors)
        outliers = lof.fit_predict(reshaped_data)
        median_value = np.median(data[outliers == 1])
        data[outliers == -1] = median_value
        logging.debug("Выбросы удалены (LOF).")
        return data

    def decompose_and_remove_outliers(self, data: pd.Series, interval: str, method='isolation_forest', contamination=None):
        decomposer = TrendDecomposer(interval) # todo: как грамотно расположить интервал
        trend, seasonal, residual = decomposer.decompose_series(data) # method='STL' ?
        if method == 'isolation_forest':
            residual_cleaned = self.remove_outliers_with_isolation_forest(residual, contamination)
        elif method == 'lof':
            residual_cleaned = self.remove_outliers_with_lof(residual, contamination)
        else:
            raise ValueError("Unsupported method for outlier removal.")
        cleaned_series = trend + seasonal + residual_cleaned
        logging.debug("Удаление выбросов через декомпозицию завершено.")
        return cleaned_series

    def remove_outliers(self, data: pd.Series, interval: str, method='combined', contamination=None):
        logging.debug("Удаление выбросов, выбран метод: %s", method)
        if method == 'combined':
            remove_outliers_method = 'isolation_forest' if len(data) > 1000 else 'lof'
            return self.decompose_and_remove_outliers(data.copy(), interval, method=remove_outliers_method,
                                                      contamination=contamination)
        elif method == 'isolation_forest':
            return self.remove_outliers_with_isolation_forest(data.copy(), contamination)
        elif method == 'lof':
            return self.remove_outliers_with_lof(data.copy(), contamination)
        else:
            raise ValueError("Unsupported outlier removal method.")