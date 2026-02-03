class ModelRegistry:
    """
        This class is responsible for model registration. For example, it can maintain a dictionary of models used in the project, with the ability to add new models.
    """
    def __init__(self):
        self.models = {}

    def register_model(self, model_name, model):
        self.models[model_name] = model

    def get_model(self, model_name):
        return self.models.get(model_name)




########################
"""
The registry stores models in a dictionary and allows registering a new model with a unique name. 
You can retrieve a registered model at any time for further work.
"""
"""
Purpose:

This file implements a model registry where models can be registered, saved, and retrieved. 
It allows centralized management of models, their state, and metadata.

Why it is needed:

Centralized model management: Registering models in the registry allows managing their state in one place. 
This can be useful for monitoring, deployment, and subsequent use of models.

Advantages for deployment: 
Models registered in the registry can be easily retrieved and used for further prediction or saving.

"""