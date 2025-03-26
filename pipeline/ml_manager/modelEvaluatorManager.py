import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import math
import logging


class ModelEvaluator:
    """
        A flexible class for evaluating machine learning models, including time series forecasting.

        Attributes:
            model: Trained model to be evaluated.
            task_type (str): Type of task ('regression' or 'classification').
            scaler: Optional scaler object for inverse transformation of predictions and true values.
    """

    __PRIME_ASSESSMENT = "smape"

    def __init__(self, model, task_type='regression', scaler=None):
        """
            Initialize the evaluator with the model, task type, and optional scaler.

            Args:
                model: A trained machine learning model.
                task_type (str): Task type, either 'regression' or 'classification'.
                scaler: Optional scaler object used for data normalization.
        """
        self.model = model
        self.task_type = task_type
        self.scaler = scaler


    def smape(self, y_true, y_pred):
        """
            Calculate the Symmetric Mean Absolute Percentage Error (SMAPE).
            Вычисление симметричной средней абсолютной процентной ошибки (SMAPE).

            Args:
                y_true (array-like): True values.
                y_pred (array-like): Predicted values.

            Returns:
                float: SMAPE value.
        """
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


    def theils_u_statistic(self, y_true, y_pred):
        """
            Calculate Theil's U statistic for time series forecasts.
            Вычисление статистики U по Теилу для оценки прогнозов временных рядов.

            Args:
                y_true (array-like): True values.
                y_pred (array-like): Predicted values.

            Returns:
                float: Theil's U statistic.
        """
        num = np.mean(np.abs(y_pred - y_true))  # Model error
        den = np.mean(np.abs(y_true[1:] - y_true[:-1]))  # Error based on sequence (baseline)
        return num / den


    def plot_predictions(self, y_true, y_pred):
        """
            Plot true vs predicted values for visual comparison.

            Args:
                y_true (array-like): True values.
                y_pred (array-like): Predicted values.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(y_true, label="True Values", color="blue")
        plt.plot(y_pred, label="Predictions", color="orange", linestyle="--")
        plt.legend()
        plt.title("True vs Predicted Values")
        plt.xlabel("Time")
        plt.ylabel("Values")
        plt.show()


    def evaluate(self, X_test, y_test):
        """
            Evaluate the model on test data.

            Args:
                X_test (array-like): Test features.
                y_test (array-like): True test values.

            Returns:
                dict: Dictionary of evaluation metrics.
        """
        predictions = self.model.predict(X_test)

        # Reverse scaling if applicable
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

            # Optional: Plot predictions
            #self.plot_predictions(y_test, predictions)

            return {'mse': mse, 'rmse': rmse, 'r2': r2, 'mae': mae, 'smape': smape, 'theils_u': theils_u}

        elif self.task_type == 'classification':
            predictions = np.argmax(predictions, axis=1)  # For softmax outputs
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='weighted')

            return {'accuracy': accuracy, 'f1_score': f1}

        else:
            raise ValueError(f"{self.__str__()} Unsupported task type. Use 'regression' or 'classification'.")


    def add_custom_metric(self, metric_name, metric_function):
        """
            Dynamically add a custom metric to the evaluator.

            Args:
                metric_name (str): Name of the custom metric.
                metric_function (callable): Function to compute the metric.
        """
        setattr(self, metric_name, metric_function)


    def _evaluate_with_split(self, X, y, split_method, **params):
        """
            Общая функция для оценки модели с использованием различных методов разделения данных.

            Parameters:
            - X (array-like): Входные данные для обучения и тестирования модели.
            - y (array-like): Метки данных для обучения и тестирования модели.
            - split_method (callable): Метод разделения данных (например, TimeSeriesSplit или RollingWindow).
            - **params: Дополнительные параметры, специфичные для метода разделения.

            Returns:
            - score (float): Общая оценка модели.
        """
        score = 0
        for train_idx, test_idx in split_method(X, **params):
            self.model.train(X[train_idx], y[train_idx])
            score += self.evaluate(X[test_idx], y[test_idx])[self.__PRIME_ASSESSMENT]

        # Log the total score
        logging.info(f"Total evaluation score across all splits: {score}")

        # Calculate and log the normalized score (average score)
        normalized_score = score / params['split_cycles']
        logging.info(f"Normalized evaluation score: {normalized_score}")

        return normalized_score


    def evaluate_using_time_series_split(self, X, y, n_splits=5):
        """
            Оценка модели с использованием TimeSeriesSplit.
        """

        def time_series_split_method(X, **params):
            tscv = TimeSeriesSplit(n_splits=params['n_splits'])
            return tscv.split(X)

        params = {'n_splits': n_splits, 'split_cycles': n_splits}
        return self._evaluate_with_split(X, y, time_series_split_method, **params)


    def evaluate_using_rolling_window(self, X, y, window_size=12):
        """
        Оценка модели с использованием Rolling Window (скользящее окно).
        """

        def rolling_window_method(X, **params):
            n_samples = len(X)
            for start in range(0, n_samples - params['window_size']):
                train_idx = range(start, start + params['window_size'])
                test_idx = range(start + params['window_size'], start + params['window_size'] + 1)
                yield train_idx, test_idx

        params = {'window_size': window_size, 'split_cycles': math.floor(len(X)/window_size)}
        return self._evaluate_with_split(X, y, rolling_window_method, **params)


    def evaluate_using_expanding_window(self, X, y):
        """
        Оценка модели с использованием Expanding Window (расширяющееся окно).
        """

        def expanding_window_method(X, **params):
            for end in range(2, len(X)):
                train_idx = range(0, end - 1)
                test_idx = [end - 1]
                yield train_idx, test_idx

        params = {'split_cycles': len(X) - 1}
        return self._evaluate_with_split(X, y, expanding_window_method, **params)


    def evaluate_using_walk_forward_validation(self, X, y, n_splits=5):
        """
        Оценка модели с использованием Walk-Forward Validation.
        """

        def walk_forward_method(X, **params):
            n_samples = len(X)
            split_size = n_samples // params['n_splits']
            for i in range(params['n_splits']):
                train_idx = range(0, split_size * (i + 1))
                test_idx = range(split_size * (i + 1), split_size * (i + 2)) if (split_size * (
                            i + 2)) <= n_samples else range(split_size * (i + 1), n_samples)
                yield train_idx, test_idx

        params = {'n_splits': n_splits, 'split_cycles': n_splits}
        return self._evaluate_with_split(X, y, walk_forward_method, **params)


    def evaluate_using_backtest(self, X, y, backtest_steps=12):
        """
        Оценка модели с использованием Backtesting.
        """

        def backtest_method(X, **params):
            n_samples = len(X)
            for step in range(1, params['backtest_steps'] + 1):
                train_idx = range(0, n_samples - step)
                test_idx = [n_samples - step]
                yield train_idx, test_idx

        params = {'backtest_steps': backtest_steps, 'split_cycles': backtest_steps}
        return self._evaluate_with_split(X, y, backtest_method, **params)


    def __str__(self):
        return "[ModelEvaluator]"


