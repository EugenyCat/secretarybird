import torch
import torch.nn as nn
import torch.optim as optim
from ml_pipeline.ml_models.baseModel import BaseModel


class LSTMWithAttentionModel(BaseModel):
    """
        LSTM Model with Attention mechanism. Inherits from BaseModel and implements
        the required methods for training, prediction, and saving/loading model states.
    """


    def __init__(self, input_dim=10, hidden_dim=64, num_layers=2, output_dim=1):
        """
        Инициализация модели LSTM с механизмом внимания.

        Parameters:
        - input_dim (int): Размерность входных данных.
        - hidden_dim (int): Размерность скрытого состояния.
        - num_layers (int): Количество слоев LSTM.
        - output_dim (int): Размерность выходных данных (например, количество классов).
        """
        super(LSTMWithAttentionModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM слой
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Механизм внимания (весовые коэффициенты)
        self.attention = nn.Linear(hidden_dim, 1)

        # Линейный слой для получения итогового предсказания
        self.fc = nn.Linear(hidden_dim, output_dim)


    def get_hyperparameter_keys(self):
        """
            Returns a list of all required hyperparameters for the model.

            Returns:
                List[str]: List of hyperparameter keys.
        """
        return [
            "input_dim",
            "hidden_dim",
            "num_layers",
            "output_dim",
            "epochs",
            "learning_rate",
            "batch_size",
            "dropout_rate",
            "optimizer",
            "attention_dim"
        ]


    def forward(self, x):
        """
        Прямой проход через модель.

        Parameters:
        - x (tensor): Входные данные (размерность: batch_size x sequence_length x input_dim).

        Returns:
        - output (tensor): Предсказания модели (размерность: batch_size x output_dim).
        """
        # Пропускаем данные через LSTM
        lstm_out, _ = self.lstm(x)

        # Рассчитываем веса внимания на основе скрытых состояний
        attention_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)

        # Контекстное представление, взвешенное по вниманию
        context = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)

        # Финальное предсказание
        output = self.fc(context)

        return output

    def train(self, X_train, y_train, epochs=100, learning_rate=0.001):
        """
        Обучение модели.

        Parameters:
        - X_train (tensor): Входные данные для обучения.
        - y_train (tensor): Целевые значения для обучения.
        - epochs (int): Количество эпох для обучения.
        - learning_rate (float): Скорость обучения для оптимизатора.
        """
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        self.train()  # Переводим модель в режим обучения

        for epoch in range(epochs):
            optimizer.zero_grad()  # Обнуляем градиенты
            outputs = self.forward(X_train)  # Получаем предсказания
            loss = criterion(outputs, y_train)  # Рассчитываем ошибку
            loss.backward()  # Обратное распространение ошибки
            optimizer.step()  # Шаг оптимизации

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

    def predict(self, X):
        """
        Предсказание для новых данных.

        Parameters:
        - X (tensor): Входные данные для предсказания.

        Returns:
        - predictions (tensor): Предсказания модели.
        """
        self.eval()  # Переводим модель в режим инференса
        with torch.no_grad():  # Отключаем вычисление градиентов
            predictions = self.forward(X)  # Получаем предсказания
        return predictions

    def save(self, model_path):
        """
        Сохраняем состояние модели.

        Parameters:
        - model_path (str): Путь для сохранения модели.
        """
        torch.save(self.state_dict(), model_path)

    def load(self, model_path):
        """
        Загружаем состояние модели.

        Parameters:
        - model_path (str): Путь к сохраненной модели.
        """
        self.load_state_dict(torch.load(model_path))


    def __str__(self):
        return 'lstm_attention'
