import logging

from sqlalchemy import desc
from sqlalchemy.inspection import inspect

from pipeline.db_schema_manager.models.orm_models import (
    Models,
    ModelTraining,
    TSPreprocessingProperties,
)
from pipeline.helpers.db_funcs_base import DBFuncsBase


class DBFuncsORM(DBFuncsBase):
    """
    todo: change this docstring in the future when the whole class methods will be done and ready
    DBFuncsML is a subclass of DBFuncsSQL. It extends the functionality to interact with
    tables from ConfigKernel and provides methods for working with the schema of tables
    defined in pipeline.helpers.init_db.

    This class inherits all functionalities from DBFuncsSQL but adds extra methods
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
        elif hasattr(row, "__dict__"):
            return {
                column.key: getattr(row, column.key)
                for column in inspect(row).mapper.column_attrs
                if column.key not in exclude_attrs  # Exclude specified attributes
            }

        # If the row is a list of ORM objects, convert each item to a dictionary
        elif isinstance(row, list):
            return [DBFuncsORM.orm_to_dict(item, exclude_attrs) for item in row]

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
            # Extract parameters from the input dictionary with explicit KeyError handling
            try:
                model_id = params["model_id"]
            except KeyError:
                model_id = None

            try:
                model_name = params["model_name"]
            except KeyError:
                model_name = None

            try:
                training_currency = params["currency"]
            except KeyError:
                training_currency = None

            try:
                training_interval = params["interval"]
            except KeyError:
                training_interval = None

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
            training_query = self.db_sqlalchemy_session.query(ModelTraining).filter(
                ModelTraining.model_id == model.model_id
            )

            if training_currency:
                training_query = training_query.filter(
                    ModelTraining.training_currency == training_currency
                )

            if training_interval:
                training_query = training_query.filter(
                    ModelTraining.training_interval == training_interval
                )

            # Order by the training score (descending) and get the best result
            training_query = training_query.order_by(desc(ModelTraining.score)).first()

            # Convert the model and training result to dictionaries and combine them
            result_dict = self.orm_to_dict(model)
            if training_query:
                result_dict.update(self.orm_to_dict(training_query))
            else:
                # Add default values if no training result is found
                result_dict.update(
                    {
                        "training_id": None,
                        "model_params": None,
                    }
                )

            return {"status": "success", "result": result_dict}, None

        except Exception as e:
            # Handle any exceptions by returning an error status with the message
            return None, {"status": "error", "message": str(e)}

    @staticmethod
    def get_ts_preprocessing_property(session, params):
        """
        Retrieve property record from database.

        Args:
            session: Database session
            params: Dict with 'ts_id' (required) and 'property_id' (optional)

        Returns:
            TSPreprocessingProperties or None
        """
        try:
            ts_id = params["ts_id"]
        except KeyError:
            ts_id = None

        try:
            property_id = params["property_id"]
        except KeyError:
            property_id = None

        if not ts_id:
            return None

        try:
            base_query = session.query(TSPreprocessingProperties).filter(
                TSPreprocessingProperties.ts_id == ts_id
            )

            # If specific property_id requested, return it directly
            if property_id:
                return base_query.filter(
                    TSPreprocessingProperties.property_id == property_id
                ).first()

            # Try to get active record first
            active_record = (
                base_query.filter(TSPreprocessingProperties.is_active == 1)
                .order_by(TSPreprocessingProperties.calculated_at.desc())
                .first()
            )

            if active_record:
                return active_record

            # Get most recent record and activate it
            latest_record = base_query.order_by(
                TSPreprocessingProperties.calculated_at.desc()
            ).first()

            if latest_record:
                latest_record.is_active = 1
                session.commit()
                logging.info(
                    f"Activated record {latest_record.property_id} for ts_id: {ts_id}"
                )

            return latest_record

        except Exception as e:
            logging.error(f"Failed to get property record for ts_id {ts_id}: {e}")
            session.rollback()
            return None

    @staticmethod
    def find_ts_preprocessing_property_by_ids(session, params):
        """
        Find existing TSPreprocessingProperties by ts_id and property_id.

        Args:
            session: Database session
            params: Dict with 'ts_id' and 'property_id'

        Returns:
            TSPreprocessingProperties or None
        """
        try:
            ts_id = params["ts_id"]
        except KeyError:
            ts_id = None

        try:
            property_id = params["property_id"]
        except KeyError:
            property_id = None

        if not ts_id or not property_id:
            return None

        try:
            return (
                session.query(TSPreprocessingProperties)
                .filter(
                    TSPreprocessingProperties.ts_id == ts_id,
                    TSPreprocessingProperties.property_id == property_id,
                )
                .first()
            )
        except Exception as e:
            logging.error(f"Failed to find TSPreprocessingProperties: {e}")
            return None

    @staticmethod
    def deactivate_active_ts_preprocessing_properties(session, params):
        """
        Deactivate all active TSPreprocessingProperties for ts_id.

        Args:
            session: Database session
            params: Dict with 'ts_id'

        Returns:
            int: Number of deactivated records
        """
        try:
            ts_id = params["ts_id"]
        except KeyError:
            ts_id = None

        if not ts_id:
            return 0

        try:
            return (
                session.query(TSPreprocessingProperties)
                .filter(
                    TSPreprocessingProperties.ts_id == ts_id,
                    TSPreprocessingProperties.is_active == 1,
                )
                .update({"is_active": 0})
            )
        except Exception as e:
            logging.error(f"Failed to deactivate properties for ts_id {ts_id}: {e}")
            session.rollback()
            return 0

    @staticmethod
    def create_ts_preprocessing_property(session, params):
        """
        Create new TSPreprocessingProperties record.

        Args:
            session: Database session
            params: Dict with all property fields

        Returns:
            TSPreprocessingProperties or None
        """
        try:
            new_record = TSPreprocessingProperties(**params)
            session.add(new_record)
            return new_record
        except Exception as e:
            logging.error(f"Failed to create TSPreprocessingProperties: {e}")
            session.rollback()
            return None

    @staticmethod
    def update_ts_preprocessing_property(session, params):
        """
        Update existing TSPreprocessingProperties record.

        Args:
            session: Database session
            params: Dict with 'record' (TSPreprocessingProperties) and 'data' (dict with updates)

        Returns:
            TSPreprocessingProperties or None
        """
        try:
            record = params["record"]
        except KeyError:
            record = None

        try:
            data = params["data"]
        except KeyError:
            data = None

        if not record or not data:
            return None

        try:
            for key, value in data.items():
                setattr(record, key, value)
            return record
        except Exception as e:
            logging.error(f"Failed to update TSPreprocessingProperties: {e}")
            session.rollback()
            return None
