"""
# TODO list

"""

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, desc
from ml_pipeline.helpers.initDB import (
    Models,
    ModelTraining
)

# Инициализация соединения с базой данных
DATABASE_URL = "clickhouse+native://username:password@host:port/default"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)


def find_model_and_best_params(model_id=None, model_name=None):
    """
    Найти модель по ID или имени и получить параметры тренировки с максимальным score.

    :param model_id: ID модели для поиска (опционально)
    :param model_name: Название модели для поиска (опционально)
    :return: Словарь с информацией о модели и лучших параметрах тренировки.
    """
    session = Session()
    try:
        # Поиск модели по model_id или model_name
        query = session.query(Models)
        if model_id:
            query = query.filter(Models.model_id == model_id)
        elif model_name:
            query = query.filter(Models.model_name == model_name)
        else:
            raise ValueError("Either 'model_id' or 'model_name' must be provided.")

        model = query.first()
        if not model:
            return {"error": "Model not found"}

        # Поиск параметров тренировки с максимальным score
        best_training = (
            session.query(ModelTraining)
            .filter(ModelTraining.model_id == model.model_id)
            .order_by(desc(ModelTraining.score))
            .first()
        )

        if not best_training:
            return {
                "model": {
                    "model_id": model.model_id,
                    "model_name": model.model_name,
                    "hyperparam_space": model.hyperparam_space,
                    "created_at": model.created_at,
                    "updated_at": model.updated_at,
                },
                "error": "No training records found for this model"
            }

        # Формирование результата
        return {
            "model": {
                "model_id": model.model_id,
                "model_name": model.model_name,
                "hyperparam_space": model.hyperparam_space,
                "created_at": model.created_at,
                "updated_at": model.updated_at,
            },
            "best_training": {
                "training_id": best_training.training_id,
                "model_params": best_training.model_params,
                "score": best_training.score,
                "created_at": best_training.created_at,
                "training_duration": best_training.training_duration,
            }
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        session.close()
