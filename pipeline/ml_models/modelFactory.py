from pipeline.ml_models.lstmWithAttentionModel.LSTMWithAttentionModel import LSTMWithAttentionModel
import logging


class ModelFactory:
    """
        A factory class for creating machine learning models based on the specified type.

        This class uses the Factory Method design pattern to create and manage different types of models.
        Adding new models is straightforward: simply register the model class in the `model_registry` dictionary.

        Example:
            model = ModelFactory.create_model("lstm_attention", input_dim=128, hidden_dim=64)

        Benefits:
        - **Scalability**: Adding new models requires minimal code changes.
        - **Encapsulation**: Creation details of each model are abstracted, making the interface cleaner.
        - **Flexibility**: Dynamically supports a variety of models based on configuration.

        Methods:
        - `create_model`: Creates a model based on the specified type and parameters.
        - `get_supported_models`: Returns a list of all supported model types.
        - `is_model_supported`: Checks if a specific model type is supported.
    """
    # Registry of model types
    __model_registry = {
        "lstm_attention": LSTMWithAttentionModel,
        # Add other model classes here
    }

    @classmethod
    def create_model(cls, model_type, **kwargs):
        """
            Creates and returns an instance of the specified model type.

            Args:
                model_type (str): The type of the model to create (e.g., 'lstm_attention').
                **kwargs: Additional arguments required for initializing the model.

            Returns:
                object: An instance of the requested model.

            Raises:
                ValueError: If the specified model type is not supported.
        """
        model_class = cls.__model_registry.get(model_type)
        if not model_class:
            raise ValueError(
                f"Unknown model type: '{model_type}'. "
                f"Use 'get_supported_models()' to see available options."
            )

        # Log model creation
        logging.info(f"{cls.__str__()} Creating model: {model_type} with parameters: {kwargs}")

        # Create and return the model instance
        return model_class(**kwargs)


    @classmethod
    def get_supported_models(cls):
        """
            Returns a list of supported model types.

            Returns:
                list: A list of strings representing the supported model types.
        """
        return list(cls.__model_registry.keys())


    @classmethod
    def is_model_supported(cls, model_type):
        """
            Checks if a specific model type is supported by the factory.

            Args:
                model_type (str): The model type to check.

            Returns:
                bool: True if the model type is supported, False otherwise.
        """
        return model_type in cls.__model_registry


    @classmethod
    def __str__(cls):
        return "[ModelFactory]"