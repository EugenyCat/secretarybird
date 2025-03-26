
# -----------------------------------------------------------------------------
# Класс для анализа временного ряда
# TimeSeriesAnalyzer
# -----------------------------------------------------------------------------

import logging
from statsmodels.tsa.stattools import adfuller, kpss
import numpy as np
import pandas as pd
from scipy.stats import zscore, probplot


class TimeSeriesAnalyzer:
    def analyze_series(self, data: pd.Series) -> dict:
        logging.debug("Начало анализа временного ряда.")
        analysis = {}
        analysis['missing_values'] = data.isnull().sum()
        # Расчёт z-score (альтернативой scipy.stats.zscore)
        z_scores = zscore(data)
        analysis['zscore_outliers'] = int((np.abs(z_scores) > 3).sum())
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        analysis['iqr_outliers'] = int(((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))).sum())
        median_val = data.median()
        mad = np.median(np.abs(data - median_val))
        modified_z = 0.6745 * (data - median_val) / mad if mad != 0 else np.zeros_like(data)
        analysis['mad_outliers'] = int((np.abs(modified_z) > 3.5).sum())
        adf_result = adfuller(data)
        analysis['adf_pvalue'] = adf_result[1]
        stationary_adf = adf_result[1] < 0.05
        try:
            kpss_result = kpss(data.dropna(), regression='c', nlags="auto")
            analysis['kpss_pvalue'] = kpss_result[1]
            stationary_kpss = kpss_result[1] > 0.05
        except Exception as e:
            analysis['kpss_pvalue'] = np.nan
            stationary_kpss = False
        window = max(30, len(data) // 10)
        rolling_mean = data.rolling(window=window, min_periods=1).mean()
        rolling_std = data.rolling(window=window, min_periods=1).std()
        mean_cv = rolling_mean.std() / rolling_mean.mean() if rolling_mean.mean() != 0 else np.inf
        std_cv = rolling_std.std() / rolling_std.mean() if rolling_std.mean() != 0 else np.inf
        analysis['rolling_mean_cv'] = mean_cv
        analysis['rolling_std_cv'] = std_cv
        autocorr_lag1 = data.autocorr(lag=1)
        analysis['lag1_autocorrelation'] = autocorr_lag1
        qq = probplot(data, dist="norm", plot=None)
        analysis['qq_correlation'] = qq[1][2]
        rolling_threshold = 0.2
        self.is_stationary = (stationary_adf and stationary_kpss and (mean_cv < rolling_threshold) and (std_cv < rolling_threshold))
        analysis['stationary'] = self.is_stationary
        logging.debug("Завершён анализ временного ряда.")
        return analysis