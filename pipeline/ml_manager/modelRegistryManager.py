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
Description and purpose:

This module is used for registering and managing model versions.
It allows saving meta-information about models, such as creation time, hyperparameters, version, etc.

Key functions:

register_model(model_name, model, metadata): Registers a new model version with metadata.

get_model(model_name, version=None): Gets a specific model version (if version is not specified, returns the latest).

list_models(): Returns a list of all registered models and their meta-information.

How it interacts with other modules:

Used by modelDeployerManager to get current models during deployment.
Works with modelTrainerManager to register new models after successful training.
"""