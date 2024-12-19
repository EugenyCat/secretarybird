from sqlalchemy import Column, String, Float, DateTime, Integer, BigInteger, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from system_files.constants.constants import CONFIG_PARAMETERS_SCHEMA_NAME

Base = declarative_base()


class Models(Base):
    __tablename__ = 'models'
    __table_args__ = (
        {'schema': CONFIG_PARAMETERS_SCHEMA_NAME},
    )

    model_id = Column(BigInteger, primary_key=True, comment="Unique identifier for the model")
    model_name = Column(String, nullable=False, comment="Name of the model (e.g., 'RandomForest', 'XGBoost')")
    hyperparam_space = Column(String, comment="JSON or string containing the model's hyperparameters")
    created_at = Column(DateTime, nullable=False, comment="Timestamp when the model entry was created")
    updated_at = Column(DateTime, nullable=False, comment="Timestamp when the model entry was last updated")

    # One-to-Many relationship: one model can have multiple training records
    trainings = relationship("ModelTraining", back_populates="model", cascade="all, delete-orphan")


class ModelTraining(Base):
    __tablename__ = 'model_training'
    __table_args__ = (
        {'schema': CONFIG_PARAMETERS_SCHEMA_NAME},
    )

    training_id = Column(BigInteger, primary_key=True, comment="Unique identifier for the training")
    model_id = Column(BigInteger, ForeignKey('ConfigKernel.models.model_id'), nullable=False, comment="Foreign key referencing the model in the models table")
    model_params = Column(String, comment="Model parameters, can be stored in JSON format")
    score = Column(Float, comment="Model quality metric (e.g., accuracy, F1 score, etc.)")
    created_at = Column(DateTime, nullable=False, comment="Timestamp when the training started")
    training_duration = Column(Integer, comment="Training duration in seconds")
    training_currency = Column(String, default='', comment="Currency used during training (e.g., USD, EUR)")
    training_interval = Column(String, default='', comment="Training interval (e.g., 1h, 3h, 1d)")

    # Relationship: training belongs to one model
    model = relationship("Models", back_populates="trainings")