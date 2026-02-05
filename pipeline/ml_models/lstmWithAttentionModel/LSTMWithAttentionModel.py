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
        Initialization of the LSTM model with attention mechanism.

        Parameters:
        - input_dim (int): Input data dimensionality.
        - hidden_dim (int): Hidden state dimensionality.
        - num_layers (int): Number of LSTM layers.
        - output_dim (int): Output data dimensionality (e.g., number of classes).
        """
        super(LSTMWithAttentionModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Attention mechanism (weight coefficients)
        self.attention = nn.Linear(hidden_dim, 1)

        # Linear layer for obtaining the final prediction
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
        Forward pass through the model.

        Parameters:
        - x (tensor): Input data (dimensions: batch_size x sequence_length x input_dim).

        Returns:
        - output (tensor): Model predictions (dimensions: batch_size x output_dim).
        """
        # Pass data through LSTM
        lstm_out, _ = self.lstm(x)

        # Calculate attention weights based on hidden states
        attention_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)

        # Context representation, weighted by attention
        context = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)

        # Final prediction
        output = self.fc(context)

        return output

    def train(self, X_train, y_train, epochs=100, learning_rate=0.001):
        """
        Model training.

        Parameters:
        - X_train (tensor): Input data for training.
        - y_train (tensor): Target values for training.
        - epochs (int): Number of epochs for training.
        - learning_rate (float): Learning rate for the optimizer.
        """
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        self.train()  # Set model to training mode

        for epoch in range(epochs):
            optimizer.zero_grad()  # Zero out gradients
            outputs = self.forward(X_train)  # Get predictions
            loss = criterion(outputs, y_train)  # Calculate loss
            loss.backward()  # Backpropagation of error
            optimizer.step()  # Optimization step

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

    def predict(self, X):
        """
        Prediction for new data.

        Parameters:
        - X (tensor): Input data for prediction.

        Returns:
        - predictions (tensor): Model predictions.
        """
        self.eval()  # Set model to inference mode
        with torch.no_grad():  # Disable gradient computation
            predictions = self.forward(X)  # Get predictions
        return predictions

    def save(self, model_path):
        """
        Save the model state.

        Parameters:
        - model_path (str): Path to save the model.
        """
        torch.save(self.state_dict(), model_path)

    def load(self, model_path):
        """
        Load the model state.

        Parameters:
        - model_path (str): Path to the saved model.
        """
        self.load_state_dict(torch.load(model_path))


    def __str__(self):
        return 'lstm_attention'