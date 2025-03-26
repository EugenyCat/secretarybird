from pipeline.helpers.db_funcs_etl import DBFuncsETL
from sqlalchemy import desc
from sqlalchemy.inspection import inspect
from pipeline.helpers.init_db import (
    Models,
    ModelTraining
)

class DBFuncsML(DBFuncsETL):
    """
        todo: change this docstring in the future when the whole class methods will be done and ready
        DBFuncsML is a subclass of DBFuncsETL. It extends the functionality to interact with
        tables from ConfigKernel and provides methods for working with the schema of tables
        defined in pipeline.helpers.init_db.

        This class inherits all functionalities from DBFuncsETL but adds extra methods
        to facilitate interactions with the database schema relevant to machine learning tasks.
    """

    @staticmethod
    def orm_to_dict(row, exclude_attrs=[]):
        """
            Converts an ORM result row to a dictionary. Optionally, columns can be excluded
            from the resulting dictionary by specifying them in the `exclude_attrs` list.

            Args:
                row (object): The ORM row to be converted.
                exclude_attrs (list, optional): List of column names to exclude from the dictionary.

            Returns:
                dict: The row as a dictionary with column names as keys and their values.
        """
        if not row:
            return {}

        # Check if the row is an ORM object (instance of mapped class)
        elif hasattr(row, '__dict__'):
            return {
                column.key: getattr(row, column.key)
                for column in inspect(row).mapper.column_attrs
                if column.key not in exclude_attrs # Exclude specified attributes
            }

        # If the row is a list of ORM objects, convert each item to a dictionary
        elif isinstance(row, list):
            return [DBFuncsML.orm_to_dict(item, exclude_attrs) for item in row]

        # Return an empty dictionary for unsupported row types
        else:
            return {}


    @staticmethod
    def add_entities_attrs_to_query(query, entities, attr_prefix=""):
        """
            Adds the attributes of each entity in `entities` to the given query, prefixing the
            column names with `attr_prefix`. This is useful for selecting specific attributes
            from multiple entities in a query.

            Args:
                query (sqlalchemy.orm.query.Query): The SQLAlchemy query to which columns will be added.
                entities (list): A list of entities (e.g., ORM models) whose attributes should be added.
                attr_prefix (str, optional): A prefix to add to the column names in the query.

            Returns:
                query (sqlalchemy.orm.query.Query): The updated query with added columns.
        """
        for entity in entities:
            # Add each column from the entity to the query with a prefixed column name
            for column in inspect(entity).c:
                query = query.add_columns(column.label(f"{attr_prefix}{column.name}"))
        return query


    def find_model_and_best_params(self, params: dict):
        """
            Find a model by its ID or name and merge it with the best training parameters.
            The method filters the `Models` table based on provided search parameters, then joins with the
            `ModelTraining` table to fetch the best training configuration based on the score.

            Args:
                params (dict): Dictionary containing search parameters (`model_id`, `model_name`, `currency`, `interval`).

            Returns:
                tuple: A tuple containing:
                    - A dictionary with combined model information and best training parameters (status "success").
                    - If an error occurs, an error dictionary with status "error" and message is returned.
        """
        try:
            # Extract parameters from the input dictionary
            model_id = params.get('model_id')
            model_name = params.get('model_name')
            training_currency = params.get('currency')
            training_interval = params.get('interval')

            # Build a query to filter the `Models` table based on the given parameters
            query = self.db_sqlalchemy_session.query(Models)
            if model_id:
                query = query.filter(Models.model_id == model_id)
            if model_name:
                query = query.filter(Models.model_name == model_name)

            # Execute the query to fetch the model
            model = query.first()
            if not model:
                return None, {"status": "error", "message": "Model not found."}

            # Filter `ModelTraining` records based on the found model_id and additional training parameters
            training_query = (
                self.db_sqlalchemy_session.query(ModelTraining)
                .filter(ModelTraining.model_id == model.model_id)
            )

            if training_currency:
                training_query = training_query.filter(ModelTraining.training_currency == training_currency)

            if training_interval:
                training_query = training_query.filter(ModelTraining.training_interval == training_interval)

            # Order by the training score (descending) and get the best result
            training_query = training_query.order_by(desc(ModelTraining.score)).first()

            # Convert the model and training result to dictionaries and combine them
            result_dict = self.orm_to_dict(model)
            if training_query:
                result_dict.update(self.orm_to_dict(training_query))
            else:
                # Add default values if no training result is found
                result_dict.update({
                    "training_id": None,
                    "model_params": None,
                })

            return {"status": "success", "result": result_dict}, None

        except Exception as e:
            # Handle any exceptions by returning an error status with the message
            return None, {"status": "error", "message": str(e)}