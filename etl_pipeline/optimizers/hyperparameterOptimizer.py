import torch
import torch.optim as optim
import torch.nn as nn
from etl_pipeline.ml_models.modelFactory import ModelFactory
import logging


class HyperparameterOptimizer:
    """
    Класс для оптимизации гиперпараметров модели.

    Методы:
    - optimize: оптимизация гиперпараметров с использованием GridSearch или RandomizedSearch.
    """

    def __init__(self, model_type, hyperparam_space, data, search_method="grid"):
        """
        Инициализация класса оптимизатора гиперпараметров для модели.

        Аргументы:
        - model_class (class): Класс модели (например, LSTMWithAttentionModel).
        - param_grid (dict): Пространство гиперпараметров для поиска.
        - data (dict): Данные для тренировки и тестирования, включая X_train, y_train, X_test, y_test.
        - search_method (str): Метод поиска гиперпараметров ("grid" или "random").
        """
        self.model_type = model_type
        self.hyperparam_space = hyperparam_space
        self.data = data
        self.search_method = search_method

    def optimize(self):
        """
        Оптимизация гиперпараметров модели с использованием GridSearch или RandomizedSearch.

        Возвращает:
        - best_params (dict): Наилучшие гиперпараметры.
        - best_score (float): Лучшая метрика (точность/ошибка).
        """
        best_params = None
        best_score = float('inf')

        # Подготовка данных
        X_train = self.data["X_train"]
        y_train = self.data["y_train"]
        X_test = self.data["X_test"]
        y_test = self.data["y_test"]

        # Перебор гиперпараметров
        for hidden_dim in self.param_grid['hidden_dim']:
            for num_layers in self.param_grid['num_layers']:
                for learning_rate in self.param_grid['learning_rate']:

                    # Создаем модель с текущими гиперпараметрами
                    model = ModelFactory().create_model(model_type=self.model_type, input_dim=X_train.shape[2], hidden_dim=hidden_dim,
                                             num_layers=num_layers, output_dim=y_train.shape[1])

                    # Обучение модели
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                    criterion = nn.MSELoss()

                    # Тренировка модели
                    model.train()
                    optimizer.zero_grad()
                    outputs = model(X_train)
                    loss = criterion(outputs, y_train)
                    loss.backward()
                    optimizer.step()

                    # Оценка модели
                    model.eval()
                    with torch.no_grad():
                        predictions = model(X_test)
                        test_loss = criterion(predictions, y_test).item()

                    # Обновление наилучших гиперпараметров
                    if test_loss < best_score:
                        best_score = test_loss
                        best_params = {
                            "hidden_dim": hidden_dim,
                            "num_layers": num_layers,
                            "learning_rate": learning_rate
                        }

        return best_params, best_score



    def _validate_params(self, model, params):
        """
            Validates that the provided parameters are compatible with the model.

            Args:
                model (object): The model instance implementing the `get_hyperparameter_keys` method.
                params (dict): Dictionary of hyperparameters to validate.

            Raises:
                ValueError: If some required parameters are missing.
        """
        if not hasattr(model, 'get_hyperparameter_keys'):
            raise AttributeError(f"Model '{model.__str__()}' does not implement 'get_hyperparameter_keys'.")

        required_keys = model.get_hyperparameter_keys()
        missing_keys = [key for key in required_keys if key not in params]

        if missing_keys:
            raise ValueError(f"Missing required parameters for model '{model.__str__()}': {missing_keys}")

        logging.info(f"Parameters validated successfully for model '{model.__str__()}'.")
