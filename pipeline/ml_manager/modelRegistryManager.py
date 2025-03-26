import mlflow
import mlflow.sklearn


class modelRegistryManager:
    """
        Class for managing model registry with MLFlow integration.
    """

    def __init__(self, tracking_uri="http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)

    def register_model(self, model, model_name):
        """
        Register a model with MLFlow.

        Args:
            model (BaseModel): The trained model object.
            model_name (str): The name of the model.

        Returns:
            str: Registered model version.
        """
        with mlflow.start_run():
            mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
        return f"Model {model_name} registered successfully."

    def load_model(self, model_name, version="latest"):
        """
        Load a model from the MLFlow registry.

        Args:
            model_name (str): The name of the model.
            version (str): The version of the model to load. Default is "latest".

        Returns:
            BaseModel: The loaded model.
        """
        model_uri = f"models:/{model_name}/{version}"
        return mlflow.sklearn.load_model(model_uri)


#####################
"""
Описание и назначение:

Этот модуль служит для регистрации и управления версиями моделей.
Он позволяет сохранять метаинформацию о моделях, такие как время создания, гиперпараметры, версия и т.д.

Ключевые функции:

register_model(model_name, model, metadata): Регистрирует новую версию модели с метаданными.

get_model(model_name, version=None): Получает конкретную версию модели (если версия не указана, возвращает последнюю).

list_models(): Возвращает список всех зарегистрированных моделей и их метаинформацию.

Как взаимодействует с остальными модулями:

Используется modelDeployerManager для получения текущих моделей при развёртывании.
Работает с modelTrainerManager для регистрации новых моделей после успешного обучения.
"""