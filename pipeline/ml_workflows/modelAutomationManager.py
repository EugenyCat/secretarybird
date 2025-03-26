from pipeline.helpers.setup import ConfigurationBuilder
from pipeline.helpers.db_funcs_ml import DBFuncsML
from pipeline.ml_models.modelFactory import ModelFactory
from pipeline.ml_manager.modelTrainerManager import ModelTrainer
from pipeline.ml_manager.modelEvaluatorManager import ModelEvaluator
from pipeline.ts_preprocessing.ts_preprocessor import TimeSeriesPreprocessor
from pipeline.ml_manager.modelRegistryManager import modelRegistryManager
from pipeline.optimizers.hyperparameterOptimizer import HyperparameterOptimizer
import json
import os
import logging
import pandas as pd


# TODO LIST
# todo 1) implement MLFlow

class ModelAutomationManager(ConfigurationBuilder):
    """
        A class for automating the process of model training, hyperparameter optimization,
        evaluation, saving, and inference. It integrates various project components
        to implement a complete machine learning pipeline.
    """

    __SOURCE_NAME = 'ml_timeseries'

    def __init__(self):
        # Initialize the parent classes
        ConfigurationBuilder.__init__(self)  # Call the initialization method for ConfigurationBuilder

        # Path to json file that contains settings for ML workflow
        try:
            config_path = os.getenv('JSON_EXTRACT_ML_SETTINGS')
            with open(config_path) as config_file:
                config_data = json.load(config_file)
                self.ml_params = config_data[self.__SOURCE_NAME]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"Error loading API configurations: {e}")


    def validate_parameters(self, input_params: dict) -> (dict, dict):
        """
            Validates the input parameters required for initializing the model automation process.

            Args:
                input_params (dict): Dictionary containing input parameters including:
                    - 'database': Name of the database storing the time series data.
                    - 'ts_table_name': Table name in the format 'currency_interval', e.g. 'btcusdt_1h'.
                    - 'model_name': Name of the machine learning model, e.g. 'lstm_attention'.

            Returns:
                tuple:
                    - dict: Success response if parameters are valid, otherwise None.
                    - dict: Error message if validation fails, otherwise None.
        """
        # Validate required parameters
        required_params = ['database', 'ts_table_name', 'model_name']

        # Check for missing parameters
        missing_params = [param for param in required_params if param not in input_params]
        if missing_params:
            error = {
                'status': 'error',
                'message': f"Missing parameters: {', '.join(missing_params)}",
            }
            return None, error

        # Validate and set parameters
        try:
            # Extract required parameters
            database = input_params['database'] # Database storing time series (TS) data
            ts_table_name = input_params['ts_table_name']  # Expected format: 'currency_interval'
            model_name = input_params['model_name'] # Name of the ML model

            # Extract currency and interval from the table name
            currency, interval = ts_table_name.split('_')

            # Configure the ModelAutomationManager instance
            (
                self.set_database(database)     # Set the database (storing TS data)
                .set_currency(currency)         # Set the currency derived from table name
                .set_interval(interval)         # Set the interval derived from table name
                .set_modelname(model_name)      # Set the model name

                # Initialize sqlalchemy session
                .set_sqlalchemy_session(self.clickhouse_conn.get_sqlalchemy_session())

                # Initialize client session for fetching TS data
                .set_client_session(self.clickhouse_conn.get_client_session())
            )
        except Exception as e:
            error = {
                'status': 'error',
                'message': f"Unexpected error during validation: {str(e)}",
            }
            return None, error

        # Return success response if all validations pass
        return {'status': 'success', 'message': 'Input parameters are validated'}, None


    def setup(self):
        """
            Initializes and configures the setup for the model training pipeline.

            The setup process includes:
                - Configuring database functions for fetching and preparing data.
                - Fetching the best model parameters and hyperparameter space from the database.
                - Creating the model using the best parameters or default settings.
                - Initializing components for training, evaluation, registry, and optimization.

            Returns:
                tuple:
                    - dict: Success response if setup is completed.
                    - dict: Error message if an issue occurs during the setup process.
        """
        try:
            # Initialize the DBFuncsML object for database interactions
            db_funcs = (
                DBFuncsML().set_client_session(self.db_client_session)      # Set database client session
                        .set_sqlalchemy_session(self.db_sqlalchemy_session) # Set SQLAlchemy session
                        .set_database(self.database)                        # Set database name
                        .set_currency(self.currency)                        # Set currency
                        .set_interval(self.interval)                        # Set interval
            )

            # Fetch and set data for the given currency and interval from the database
            ts_data = db_funcs.get_data()

            ts_preprocessor_manager = TimeSeriesPreprocessor(self.interval)
            X, y = ts_preprocessor_manager.process(ts_data)

            self.set_data(X, y)

            # Fetch the best model parameters and hyperparameter space from the database
            result_model_and_best_params, error = db_funcs.find_model_and_best_params(
                {
                    'model_name': self.modelname,
                    'currency': self.currency,
                    'interval': self.interval
                }
            )
            if error:
                return None, {"status": "error", "message": error['message']}

            # Extract the best parameters and hyperparameter space
            best_params = result_model_and_best_params['result']['model_params'] or {}
            if not best_params:
                logging.info(f"{self.__str__()} Best parameters not found.") # todo: improve this log

            hyperparam_space = json.loads(result_model_and_best_params['result']['hyperparam_space'])

            # Create and set the model using ModelFactory
            self.set_model(ModelFactory().create_model(self.modelname, **best_params))

            # Initialize the trainer for training the model
            self.trainer = ModelTrainer()

            # Initialize the evaluator for evaluating the model's performance
            self.evaluator = ModelEvaluator(self.model)

            # Initialize the registry manager for saving and managing models
            # todo: 1) save a model into db or use MLFlow
            #self.registry = modelRegistryManager()

            # Initialize the hyperparameter optimizer
            self.optimizer = HyperparameterOptimizer(self.modelname, hyperparam_space, self.data, self.evaluator)

        except Exception as e:
            # Handle unexpected errors during the setup process
            return None, {"status": "error", "message": f"Unexpected error during setup: {str(e)}"}

        # Return success response if setup completes successfully
        return {'status': 'success', 'result': 'Setup is completed successfully.'}, None


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

    def run(self, input_params):
        # call validate params validated_params = self.validate_params(params)
        validated_parameters_response, error = self.validate_parameters(input_params)

        if error:
            error_message = f"{self.__str__()} {error['message']}"
            logging.error(error_message)
            return None, {"status": "error", "message": error_message}

        try:
            setup_response, error = self.setup()

            if error:
                error_message = f"{self.__str__()} {error['message']}"
                logging.error(error_message)
                return None, {"status": "error", "message": error_message}

            return {"status": "success", "message": setup_response['result']}, None
        except Exception as e:
            error_message = f"{self.__str__()} {str(e)}"
            logging.error(error_message)
            return None, {"status": "error", "message": str(e)}


    def __str__(self):
        return f'<[ml_workflows/modelAutomationManager.py] {self.modelname}, {self.currency}_{self.interval}>'



# Инициализация и запуск автоматизации
#manager = ModelAutomationManager()
#manager.run({
#    "modelname": "lstm_attention",
#    "currency": "btcusdt",
#    "interval": "1h"
#})

#result = manager.automate()
#print(result)
