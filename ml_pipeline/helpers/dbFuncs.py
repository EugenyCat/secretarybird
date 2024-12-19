"""
# TODO list

"""
import logging

from ml_pipeline.helpers.ml_setup import ConfigurationBuilder
from sqlalchemy import desc
from sqlalchemy.inspection import inspect
from ml_pipeline.helpers.initDB import (
    Models,
    ModelTraining
)

class DBFuncs(ConfigurationBuilder):

    @staticmethod
    def orm_to_dict(row, exclude_attrs=[]):
        """
            Convert an ORM result row to a dictionary.
        """
        if not row:
            return {}
        elif hasattr(row, '__dict__'):  # Это объект ORM
            return {
                column.key: getattr(row, column.key)
                for column in inspect(row).mapper.column_attrs
                if column.key not in exclude_attrs
            }
        elif isinstance(row, list):  # Если это список объектов
            return [DBFuncs.orm_to_dict(item, exclude_attrs) for item in row]
        else:
            return {}


    @staticmethod
    def add_entities_attrs_to_query(query, entities, attr_prefix=""):
        for entity in entities:
            for column in inspect(entity).c:
                query = query.add_columns(column.label(f"{attr_prefix}{column.name}"))
        return query


    def find_model_and_best_params(self, params: dict):
        """
            Find a model by ID or name and merge it with the best training parameters.

            :param params: Dictionary containing search parameters (model_id or model_name).
            :return: Dictionary with combined model information and best training parameters.
        """
        try:
            # Validate input parameters
            model_id = params.get('model_id')
            model_name = params.get('model_name')
            training_currency = params.get('training_currency')
            training_interval = params.get('training_interval')

            if not model_id and not model_name:
                return {"error": "Either 'model_id' or 'model_name' must be provided."}

            # Filter the Models table
            query = self.db_session.query(Models)
            if model_id:
                query = query.filter(Models.model_id == model_id)
            if model_name:
                query = query.filter(Models.model_name == model_name)

            # Fetch the filtered model
            model = query.first()
            if not model:
                return {"error": "Model not found."}

            # Join with ModelTraining after filtering
            training_query = (
                self.db_session.query(ModelTraining)
                .filter(ModelTraining.model_id == model.model_id)
            )

            # Apply additional filters if parameters are provided
            if training_currency:
                training_query = training_query.filter(ModelTraining.training_currency == training_currency)

            if training_interval:
                training_query = training_query.filter(ModelTraining.training_interval == training_interval)

            training_query = training_query.order_by(desc(ModelTraining.score)).first()

            # Convert results to a dictionary
            result_dict = self.orm_to_dict(model)
            if training_query:
                result_dict.update(self.orm_to_dict(training_query))
            else:
                result_dict.update({
                    "training_id": None,
                    "model_params": None,
                })

            return result_dict

        except Exception as e:
            # Log the error if needed
            # logger.error(f"Error in find_model_and_best_params: {str(e)}")
            return {"error": str(e)}