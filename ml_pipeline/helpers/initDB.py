from sqlalchemy import Column, String, Float, DateTime, Integer, BigInteger
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


# Таблица models
class Models(Base):
    __tablename__ = 'models'
    __table_args__ = (
        {'schema': 'MODEL_REGISTRY', 'engine': 'MergeTree() ORDER BY (model_id)'},
    )

    model_id = Column(BigInteger, primary_key=True, comment="Уникальный идентификатор модели")
    model_name = Column(String, nullable=False, comment="Название модели (например, 'RandomForest', 'XGBoost')")
    hyperparam_space = Column(String, comment="JSON или строка с гиперпараметрами модели")
    created_at = Column(DateTime, nullable=False, comment="Дата создания записи (время начала обучения модели)")
    updated_at = Column(DateTime, nullable=False, comment="Дата последнего обновления модели")


# Таблица model_training
class ModelTraining(Base):
    __tablename__ = 'model_training'
    __table_args__ = (
        {'schema': 'MODEL_REGISTRY', 'engine': 'MergeTree() PARTITION BY model_id ORDER BY (training_id)'},
    )

    training_id = Column(BigInteger, primary_key=True, comment="Уникальный идентификатор тренировки")
    model_id = Column(BigInteger, nullable=False, comment="Ссылка на модель из таблицы models")
    model_params = Column(String, comment="Параметры модели, также можно хранить в формате JSON")
    score = Column(Float, comment="Оценка качества модели (например, метрика точности, F1 score и т.д.)")
    created_at = Column(DateTime, nullable=False, comment="Дата и время начала тренировки")
    training_duration = Column(Integer, comment="Длительность обучения в секундах")
