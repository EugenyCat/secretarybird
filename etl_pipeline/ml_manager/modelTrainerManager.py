from etl_pipeline.ml_models.modelFactory import ModelFactory

class ModelTrainer:

    def train_model(self, model_type, X_train, y_train, best_params):
        """
        Обучает модель с использованием переданных лучших гиперпараметров.

        Аргументы:
            model_type (class): Тип модели для обучения.
            X_train (array): Данные для обучения.
            y_train (array): Целевые значения для обучения.
            best_params (dict): Лучшие параметры для модели, полученные от оптимизатора.

        Возвращает:
            model: Обученную модель.
        """
        # Проверка соответствия размеров X_train и y_train
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Количество примеров в X_train и y_train должно совпадать.")

        # Создание модели
        model = ModelFactory().create_model(model_type)

        # Обучение модели
        try:
            model.train(X_train, y_train, **best_params)
        except Exception as e:
            raise RuntimeError(f"Ошибка при обучении модели: {e}")

        # Логирование информации о завершении обучения
        print(f"Модель {model_type} успешно обучена с параметрами: {best_params}")

        return model
