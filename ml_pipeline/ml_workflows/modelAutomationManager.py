import torch
from ml_pipeline.ml_models.modelFactory import ModelFactory
from ml_pipeline.ml_manager.modelTrainerManager import ModelTrainer
from ml_pipeline.ml_manager.modelEvaluatorManager import ModelEvaluator
from ml_pipeline.ml_manager.modelRegistryManager import modelRegistryManager
from ml_pipeline.optimizers.hyperparameterOptimizer import HyperparameterOptimizer

class ModelAutomationManager:
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

    def __init__(self, model_name, model_params, data, hyperparam_space):
        """
        Инициализация ModelAutomationManager с заданным классом модели, параметрами и данными.

        Аргументы:
            model_class (class): Класс модели, которую нужно создать (например, LSTMWithAttentionModel).
            model_params (dict): Параметры для инициализации модели.
            data (dict): Словарь, содержащий 'X_train', 'y_train', 'X_test' и 'y_test'.
            hyperparam_space (dict): Пространство для поиска гиперпараметров.
        """

        """
        создается модель с какими-то начальными параметрами 
        - либо использовать последние лучшие параметры лучшей модели - либо какой то дефолтный набор
        """
        # todo: как организовать передачу параметров в даге? надо понять где в каком виде будут храниться model_name, model_params, hyperparam_space
        # todo: add a method that read the db, find the model_name with the best/default params
        self.model = ModelFactory().create_model(model_name, **model_params)

        """ self.trainer у него просто метод train_model() который возвращает обученную модель (та же модель с вызванным train) """
        self.trainer = ModelTrainer()

        """ self.evaluator объект который будет оценивать качество модели """
        # todo: make more flexible for different models and more task_type
        self.evaluator = ModelEvaluator(self.model, task_type='regression')

        """ self.registry будет отвечать за сохранение модели"""
        # todo: доработать , где и как будет храниться, (бекапы ?? бекап бд будет достаточно ?), как загружать заново
        # как организовать дальнейшее взаимодействие со старыми моделями ? (отдельный интерфейс который просто берет лучшую/посл модель и получает прогноз)
        self.registry = modelRegistryManager()

        """ self.optimizer - оптимизатор который должен искать лучшие параметры """
        # todo: надо сделать Bayesian Optimization и Optuna и Ray Tune (также torch.optim, мб кто еще ) : и смотреть на практике кто оптимальнее
        # todo: там уже внутри есть ошибка что у model вызывается метод parametres - надо написать HyperparameterOptimizer корректно в соответсвт с интерфесвом ML моделей
        self.optimizer = HyperparameterOptimizer(model_name, hyperparam_space, data) # todo create flexible barian

        # просто сетка данных
        # todo: переделать в параметры типа currency interval и загружать из бд
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
manager = ModelAutomationManager(
    model_name='lstm_attention',
    model_params={"input_dim": 5, "hidden_dim": 64, "num_layers": 2, "output_dim": 1},
    data=data,
    hyperparam_space=hyperparam_space
)

result = manager.automate()
print(result)
