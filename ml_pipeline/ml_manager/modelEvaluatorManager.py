import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelEvaluator:
    """
    Класс для оценки модели LSTM с вниманием для задач прогнозирования временных рядов.
    """

    def __init__(self, model, task_type='regression', scaler=None):
        self.model = model
        self.task_type = task_type
        self.scaler = scaler  # Массив для инвертирования нормализации (если применялась)

    def smape(self, y_true, y_pred):
        """
        Вычисление симметричной средней абсолютной процентной ошибки (SMAPE).
        """
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

    def theils_u_statistic(self, y_true, y_pred):
        """
        Вычисление статистики U по Теилу для оценки прогнозов временных рядов.
        """
        num = np.mean(np.abs(y_pred - y_true))  # Ошибка модели
        den = np.mean(np.abs(y_true[1:] - y_true[:-1]))  # Ошибка на основе последовательности (baseline)
        return num / den

    def evaluate(self, X_test, y_test):
        """
        Оценка модели на тестовых данных с учетом специфики временных рядов.

        Аргументы:
            X_test (array-like): Тестовые данные.
            y_test (array-like): Истинные значения.
            task_type (str): Тип задачи (для прогнозирования временных рядов всегда 'regression').

        Возвращает:
            dict: Метрики модели.
        """
        predictions = self.model.predict(X_test)

        # Преобразование обратно в исходный масштаб, если применялась нормализация
        if self.scaler:
            predictions = self.scaler.inverse_transform(predictions)
            y_test = self.scaler.inverse_transform(y_test)

        if self.task_type == 'regression':
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            smape = self.smape(y_test, predictions)
            theils_u = self.theils_u_statistic(y_test, predictions)

            return {'mse': mse, 'rmse': rmse, 'r2': r2, 'mae': mae, 'smape': smape, 'theils_u': theils_u}

        else:
            raise ValueError("Unsupported task type. Use 'regression' for time series forecasting.")


######################
"""
Описание и назначение:

Модуль отвечает за оценку производительности моделей после обучения. 
Он позволяет вычислять метрики, такие как точность, среднеквадратичная ошибка и любые другие показатели качества работы модели.

Ключевые функции:

evaluate(model, X_test, y_test): Выполняет прогнозы модели на тестовых данных и вычисляет метрики.

log_metrics(metrics): Логирует метрики для последующего анализа.

Как взаимодействует с остальными модулями:

Использует обученные модели из ml_models.

Может взаимодействовать с инструментами логирования или мониторинга, 
например, для записи метрик в базу данных или их отображения в Grafana.
"""