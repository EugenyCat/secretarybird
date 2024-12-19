import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
import matplotlib.pyplot as plt


class ModelEvaluator:
    """
        A flexible class for evaluating machine learning models, including time series forecasting.

        Attributes:
            model: Trained model to be evaluated.
            task_type (str): Type of task ('regression' or 'classification').
            scaler: Optional scaler object for inverse transformation of predictions and true values.
    """

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


    def __str__(self):
        return "[ModelEvaluator]"