import optuna
import logging


class HyperparameterOptimizer:
    def __init__(self, model, hyperparam_space, data, evaluator):
        """
        Инициализация оптимизатора гиперпараметров.

        Parameters:
        - model (BaseModel): Модель, для которой выполняется оптимизация гиперпараметров.
        - hyperparam_space (dict): Пространство гиперпараметров.
        - data (dict): Данные, необходимые для обучения модели.
        """
        self.model = model
        self.hyperparam_space = hyperparam_space
        self.data = data
        self.evaluator = evaluator
        self.best_params = None


    def objective(self, trial):
        """
        Функция цели для Optuna, которая будет вызываться для каждой итерации.

        Parameters:
        - trial (optuna.trial.Trial): Экземпляр Optuna Trial, который используется для оптимизации.

        Returns:
        - float: Значение функции потерь (или метрики, которую мы минимизируем).
        """
        # Генерация гиперпараметров для модели на основе пространства поиска
        params = self._get_params_for_trial(trial)

        # Обучение модели с этими гиперпараметрами
        self.model.train(self.data['X'], self.data['y'], **params)

        # Выбор Optuna одного из методов валидации для использования в текущем прогоне.
        validation_method = trial.suggest_categorical('validation_type', [
            self.evaluator.evaluate_using_time_series_split,
            self.evaluator.evaluate_using_rolling_window,
            self.evaluator.evaluate_using_expanding_window,
            self.evaluator.evaluate_using_walk_forward_validation,
            self.evaluator.evaluate_using_backtest
        ])

        # Оценка качества модели # todo: check the result
        score = validation_method(self.data['X'], self.data['y'])

        # Логирование значений потерь
        logging.info(f"Trial: {trial.number}, Score: {score}")

        return score


    def _get_params_for_trial(self, trial):
        """
            Генерация гиперпараметров для каждой итерации на основе пространства гиперпараметров.

            Parameters:
            - trial (optuna.trial.Trial): Экземпляр Optuna Trial.

            Returns:
            - dict: Словарь с гиперпараметрами для этой итерации.
        """
        params = {}
        for key in self.hyperparam_space:
            params[key] = trial.suggest_categorical(key, self.hyperparam_space[key])
        return params


    def evaluate_model(self, X_test, y_test, assessment="smape"):
        """
            Оценка модели на валидационных данных.

            Parameters:
            - X_test (tensor): Входные данные для валидации.
            - y_test (tensor): Целевые значения для валидации.

            Returns:
            - float: Значение assessment.
        """
        evaluation = self.evaluator.evaluate(X_test, y_test)[assessment]
        return evaluation


    def optimize(self, n_trials=50):
        """
            Запуск оптимизации гиперпараметров с использованием Optuna.

            Parameters:
            - n_trials (int): Количество проб (итераций) для оптимизации.
        """
        study = optuna.create_study(direction='minimize')  # Оптимизируем для минимизации потерь
        study.optimize(self.objective, n_trials=n_trials)

        self.best_params = study.best_params
        logging.info(f"Best hyperparameters: {self.best_params}")
        return self.best_params


    def get_best_params(self):
        """
            Возвращает наилучшие гиперпараметры после оптимизации.

            Returns:
            - dict: Наилучшие параметры.
        """
        return self.best_params
