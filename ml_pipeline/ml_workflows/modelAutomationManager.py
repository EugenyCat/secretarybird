from ml_pipeline.helpers.ml_setup import ConfigurationBuilder
from system_files.constants.constants import JSON_EXTRACT_ML_SETTINGS
from ml_pipeline.helpers.dbFuncs import DBFuncs
import json

import torch
from ml_pipeline.ml_models.modelFactory import ModelFactory
from ml_pipeline.ml_manager.modelTrainerManager import ModelTrainer
from ml_pipeline.ml_manager.modelEvaluatorManager import ModelEvaluator
from ml_pipeline.ml_manager.modelRegistryManager import modelRegistryManager

from ml_pipeline.optimizers.hyperparameterOptimizer import HyperparameterOptimizer

import logging

class ModelAutomationManager(ConfigurationBuilder):
    """
        Класс для автоматизации процесса обучения, оптимизации гиперпараметров,
        оценки, сохранения и предсказания с использованием заданной модели (например, LSTMWithAttentionModel).
        Интегрирует различные компоненты проекта для реализации полного пайплайна.

        Атрибуты:
            model (BaseModel): Модель машинного обучения, которую нужно автоматизировать (например, LSTMWithAttentionModel).
            trainer (ModelTrainerManager): Управляет обучением модели.
            evaluator (ModelEvaluatorManager): Управляет оценкой модели.
            registry (ModelRegistryManager): Управляет регистрацией и сохранением модели.
            optimizer (HyperparameterOptimizer): Оптимизирует гиперпараметры модели.
    """

    __SOURCE_NAME = 'ml_timeseries'

    def __init__(self):
        # Initialize the parent classes
        ConfigurationBuilder.__init__(self)  # Call the initialization method for ConfigurationBuilder

        # path to json file that contains settings for etl
        self.load_params = json.load(
            open(JSON_EXTRACT_ML_SETTINGS, 'r')
        )[self.__SOURCE_NAME]


    def setup(self):
        # , model_name, model_params, data, hyperparam_space
        """
            Инициализация ModelAutomationManager с заданным классом модели, параметрами и данными.

            Аргументы:
                model_class (class): Класс модели, которую нужно создать (например, LSTMWithAttentionModel).
                model_params (dict): Параметры для инициализации модели.
                data (dict): Словарь, содержащий 'X_train', 'y_train', 'X_test' и 'y_test'.
                hyperparam_space (dict): Пространство для поиска гиперпараметров.
        """

        self.set_database(self.load_params['database'])
        self.set_db_session(self.clickhouse_conn.get_sqlalchemy_session(self.database))

        db_funcs = DBFuncs().set_db_session(self.db_session)

        result_model_and_best_params = db_funcs.find_model_and_best_params({'model_name': self.modelname})
        logging.warning(f'!!! {result_model_and_best_params}')
        best_params = result_model_and_best_params['model_params'] or {}
        hyperparam_space = json.loads(result_model_and_best_params['hyperparam_space'])

        self.set_model(ModelFactory().create_model(self.modelname, **best_params))

        self.trainer = ModelTrainer()

        self.evaluator = ModelEvaluator(self.model)

        """ self.registry будет отвечать за сохранение модели"""
        self.registry = modelRegistryManager()



        """ self.optimizer - оптимизатор который должен искать лучшие параметры """
        # todo: надо сделать Bayesian Optimization и Optuna и Ray Tune (также torch.optim, мб кто еще ) : и смотреть на практике кто оптимальнее
        # todo: там уже внутри есть ошибка что у model вызывается метод parametres - надо написать HyperparameterOptimizer корректно в соответсвт с интерфесвом ML моделей
        self.optimizer = HyperparameterOptimizer(self.modelname, hyperparam_space, data) # todo create flexible barian

        return 0

        # просто сетка данных
        # todo: переделать в параметры типа currency interval и загружать из бд
        # в базе данных добавить в таблицу models_training поле данные на которых было обучение "BTCUSDT_1h" и именно по нему делать поиск лучших
        # создать функционал который загрузит данные эти
        # идея что модели будут под конкретные данные
        self.data = data

    def automate(self):
        """
        Выполняет полный процесс автоматизации: оптимизация гиперпараметров, обучение модели,
        оценка, сохранение и предсказания.
        """
        print("Начало оптимизации гиперпараметров...")
        best_params = self.optimizer.optimize()

        # Обновление модели с наилучшими параметрами
        self.model = self.model.__class__(**best_params)

        print("Обучение модели...")
        self.trainer.train_model(self.model, self.data['X_train'], self.data['y_train'])

        print("Оценка модели...")
        metrics = self.evaluator.evaluate(self.data['X_test'], self.data['y_test'])
        print(f"Метрики оценки: {metrics}")

        print("Сохранение модели...")
        model_id = self.registry.register_model(self.model) # сюда же best_params, metrics

        print("Получение предсказаний...")
        predictions = self.model.predict(self.data['X_test'])
        print(f"Предсказания: {predictions}")

        return {
            "model_id": model_id,
            "best_params": best_params,
            "metrics": metrics,
            "predictions": predictions
        }

    def run(self, params):
        # call validate params validated_params = self.validate_params(params)
        validated_params = params

        (
            self.set_modelname(validated_params['modelname'])
            .set_currency(validated_params['currency'])
            .set_interval(validated_params['interval'])
        )

        self.setup()

# Пример использования:
data = {
    "X_train": torch.randn(100, 10, 5),
    "y_train": torch.randn(100, 1),
    "X_test": torch.randn(20, 10, 5),
    "y_test": torch.randn(20, 1)
}

# Пример пространства для поиска гиперпараметров
hyperparam_space = {
    "hidden_dim": [32, 64, 128],
    "num_layers": [1, 2, 3],
    "learning_rate": [0.001, 0.01]
}

# Инициализация и запуск автоматизации
manager = ModelAutomationManager()
manager.run({
    "modelname": "lstm_attention",
    "currency": "btcusdt",
    "interval": "1h"
})

#result = manager.automate()
#print(result)
