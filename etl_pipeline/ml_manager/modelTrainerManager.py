import logging

class ModelTrainer:
    """
        A class responsible for training machine learning models.
    """

    def train_model(self, model, X_train, y_train, best_params):
        """
            Trains a machine learning model using the provided training data and best hyperparameters.

            Args:
                model (object): The model to be trained (e.g., <class 'etl_pipeline.ml_models.lstmWithAttentionModel).
                X_train (array-like): Features for training the model.
                y_train (array-like): Target values for training the model.
                best_params (dict): Best hyperparameters for the model.

            Returns:
                model: The trained model.

            Raises:
                ValueError: If the dimensions of X_train and y_train do not match.
                RuntimeError: If an error occurs during model training.
        """
        # Validate input dimensions
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"{self.__str__()} The number of samples in X_train and y_train must match.")


        # Train the model
        try:
            model.train(X_train, y_train, **best_params)
            logging.info(f"Model '{model.__str__()}' trained successfully with parameters: {best_params}")
        except Exception as e:
            logging.error(f"Error during training of model '{model.__str__()}': {e}")
            raise RuntimeError(f"Error during model training: {e}")

        return model


    def __str__(self):
        return "[ModelTrainer]"