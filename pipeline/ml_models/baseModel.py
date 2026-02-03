from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
        This file contains the base class for all models, which will implement common methods such as:

        train() — for training the model.
        predict() — for making predictions.
        save() — for saving the model.
        load() — for loading the model.
    """
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    @abstractmethod
    def save(self, model_path):
        pass

    @abstractmethod
    def load(self, model_path):
        pass


class NeuralNetworkModel(BaseModel):
    """
        Additional methods for neural networks.
    """
    @abstractmethod
    def train_on_batch(self, X_batch, y_batch):
        pass

    @abstractmethod
    def get_layers(self):
        pass





##################################
"""
Here BaseModel defines the structure for all model classes, requiring 
them to implement the train(), predict(), save(), and load() methods.
"""
"""
This file contains the base class that provides a common interface for all machine learning models. 
This ensures that all models, regardless of their type, will have the same basic methods for training, 
prediction, saving, and loading.

Why it is needed:

Abstraction: All models, including classical algorithms and neural networks, must follow the same interface 
for unified usage. This allows them to be used in the same context (for example, in a pipeline) 
without needing to know which model is being used.

This simplifies adding new models, as each new model will simply implement these standard methods.

"""

#   -> baseModel.py defines the interface for all models.
#   -> modelFactory.py allows creating various models.
#   -> modelRegistry.py manages all models centrally.
#   -> lstmWithAttentionModel/ contains the code for a specific LSTM with Attention model.