"""
N-BEATS time series decomposition method.

Implements Neural Basis Expansion Analysis Time Series for interpretable
decomposition through neural network architecture with trend and seasonal stacks.
Optimal for: data_length > 1000 AND baseline_quality < 0.4 AND data_quality > 0.8
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
from torch.utils.data import DataLoader, TensorDataset  # type: ignore

from pipeline.helpers.utils import validate_required_locals
from pipeline.timeSeriesProcessing.decomposition.methods.baseDecomposerMethod import (
    BaseDecomposerMethod,
)

version = "1.0.0"


# Data requirements constants
MIN_DATA_LENGTH = 1000  # Minimum data points for effective neural network training
MAX_BASELINE_QUALITY = (
    0.4  # Maximum baseline quality for N-BEATS usage (nonlinear patterns)
)
MIN_DATA_QUALITY = 0.8  # Minimum data quality required for clean neural training

# Training validation constants
MIN_INPUT_SIZE = 10  # Minimum input size for meaningful N-BEATS training
MIN_FORECAST_SIZE = 1  # Minimum forecast size

# Device detection constants
GPU_TEST_VALUE = 1.0  # Test value for GPU tensor creation
MEMORY_CONVERSION_FACTOR = 1024 * 1024  # Bytes to MB conversion

# Data preprocessing constants
FLOAT32_DTYPE = np.float32  # Data type for neural network processing
STD_ZERO_FALLBACK = 1.0  # Fallback std when normalization fails

# Training loop constants
INITIAL_LOSS = 0.0  # Initial loss value
BATCH_COUNT_INITIAL = 0  # Initial batch count

# Progress logging constants
PROGRESS_LOG_DIVISOR = 10  # Log every 1/10th of total epochs
MIN_PROGRESS_LOG_INTERVAL = 1  # Minimum interval between progress logs

# Early stopping constants
EARLY_STOPPING_LOSS_THRESHOLD = 0.001  # Loss threshold for early stopping
EARLY_STOPPING_MIN_IMPROVEMENT = 1e-6  # Minimum improvement to continue training
EARLY_STOPPING_PATIENCE_EPOCHS = 5  # Epochs to wait without improvement

# Convergence constants
CONVERGENCE_LOSS_THRESHOLD = 1.0  # Loss threshold for convergence check
CONVERGENCE_STABILITY_EPOCHS = 3  # Epochs to check for loss stability

# Component reconstruction constants
GENERIC_SPLIT_RATIO = 0.5  # Split ratio for generic stack components

# Memory management constants
GPU_MEMORY_THRESHOLD = 0  # Threshold for GPU memory monitoring

# Model architecture defaults (non-adaptive)
DEFAULT_NUM_BLOCKS = 3
DEFAULT_NUM_LAYERS = 4
DEFAULT_POLYNOMIAL_DEGREE = 3


class NBEATSDecomposerMethod(BaseDecomposerMethod):
    """
    N-BEATS (Neural Basis Expansion Analysis Time Series) decomposition.

    Neural network architecture for interpretable decomposition
    of time series with extraction of trend and seasonal components.
    Uses interpretable trend/seasonal stacks with polynomial
    and trigonometric bases.

    Mathematical requirements:
    - data_length > 1000 (for effective neural network training)
    - baseline_quality < 0.4 (nonlinear patterns)
    - data_quality > 0.8 (clean data for training)

    References:
        Oreshkin et al. "N-BEATS: Neural basis expansion analysis for
        interpretable time series forecasting" (ICLR 2020)
    """

    DEFAULT_CONFIG = {
        # Architecture constants
        "fallback_on_error": True,  # Fallback on training errors (Does not require adaptation)
        "num_blocks": DEFAULT_NUM_BLOCKS,  # Number of blocks in each stack (Does not require adaptation)
        "num_layers": DEFAULT_NUM_LAYERS,  # Number of layers in block (Does not require adaptation)
        "polynomial_degree": DEFAULT_POLYNOMIAL_DEGREE,  # Degree of polynomial basis for trend (Does not require adaptation)
        "device": None,  # Compute device [auto-detect] (Does not require adaptation)
        # Correctly adapting parameters
        # "input_size": None,               # Input sequence size (adapted in configDecomposition)
        # "forecast_size": None,            # Forecast sequence size (adapted in configDecomposition)
        # "stack_types": None,              # Neural stack types [trend/seasonality/generic] (adapted in configDecomposition)
        # "epochs": None,                   # Number of training epochs (adapted in configDecomposition)
        # "batch_size": None,               # Batch size for training (adapted in configDecomposition)
        # "harmonics": None,                # Number of harmonics for seasonal stack (adapted in configDecomposition)
        # "layer_width": None,              # Width of hidden neural network layers (adapted in configDecomposition)
        # "learning_rate": None,            # Learning rate for optimizer (adapted in configDecomposition)
        # "early_stopping_patience": None,  # Patience for early stopping (adapted in configDecomposition)
        # "gradient_clipping": None,        # Gradient clipping for stability (adapted in configDecomposition)
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize N-BEATS method with fail-fast PyTorch validation.

        Args:
            config: Configuration (must be fully adapted)

        Raises:
            ImportError: If PyTorch is unavailable
            ValueError: If configuration is invalid
        """
        # Merge configuration with defaults
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged_config)

        # Device detection and setup
        self.device = self._detect_device()
        logging.info(f"{self} - Using device: {self.device}")

    def __str__(self) -> str:
        """Standard string representation for logging."""
        return (
            f"NBEATSDecomposer(stacks={self.config.get('stack_types', [])}, "
            f"device={getattr(self, 'device', 'unknown')})"
        )

    def _validate_config(self) -> None:
        """Validate N-BEATS specific parameters."""
        # All parameters must be adapted by configDecomposition.py
        required_params = [
            "input_size",
            "forecast_size",
            "stack_types",
            "epochs",
            "batch_size",
        ]

        validate_required_locals(required_params, self.config)

        # Validate parameter values
        if self.config["input_size"] < MIN_INPUT_SIZE:
            raise ValueError(
                f"input_size must be >= {MIN_INPUT_SIZE} for meaningful N-BEATS training"
            )

        if self.config["forecast_size"] < MIN_FORECAST_SIZE:
            raise ValueError(
                f"forecast_size must be >= {MIN_FORECAST_SIZE}"
            )

        if not isinstance(self.config["stack_types"], list):
            raise ValueError("stack_types must be a list")

        valid_stack_types = {"trend", "seasonality", "generic"}
        for stack_type in self.config["stack_types"]:
            if stack_type not in valid_stack_types:
                raise ValueError(
                    f"Invalid stack_type '{stack_type}'. "
                    f"Must be one of: {valid_stack_types}"
                )

    def process(
        self, data: pd.Series, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform N-BEATS time series decomposition.

        Args:
            data: Time series for decomposition
            context: Processing context

        Returns:
            Standardized decomposition result
        """
        try:
            # 1. CRITICAL fail-fast validation
            critical_validation = self.validate_input_critical(data)
            if critical_validation is not None:
                return critical_validation

            # 2. Standard input data validation
            validation = self.validate_input(data, min_length=self.config["input_size"])
            if validation["status"] == "error":
                return validation

            # 2. Extract context
            context_params = self.extract_context_parameters(context)

            # 3. Data preprocessing
            processed_data = self.preprocess_data(data)

            logging.info(
                f"{self} - Starting N-BEATS decomposition: "
                f"data_length={len(processed_data)}, "
                f"input_size={self.config['input_size']}, "
                f"forecast_size={self.config['forecast_size']}"
            )

            # 4. Check sufficient data for training
            min_required = self.config["input_size"] + self.config["forecast_size"]
            if len(processed_data) < min_required:
                return self._fallback_decomposition(
                    processed_data, context_params, "insufficient_data_for_training"
                )

            # 5. N-BEATS algorithmic logic
            trend, seasonal, residual, model_info = self._train_and_decompose(
                processed_data
            )

            # 6. Prepare additional data
            additional_data = {
                "model_type": "pytorch_nbeats",
                "stack_types": self.config["stack_types"],
                "input_size": self.config["input_size"],
                "forecast_size": self.config["forecast_size"],
                "device_used": str(self.device),
                "total_parameters": model_info.get("total_parameters", 0),
                "training_loss": model_info.get("final_loss", float("inf")),
                "training_sequences": model_info.get("training_sequences", 0),
            }

            # 7. Result through base method
            return self.prepare_decomposition_result(
                trend, seasonal, residual, data, context_params, additional_data
            )

        except Exception as e:
            return self.handle_error(e, "N-BEATS decomposition")

    def _detect_device(self) -> Any:
        """
        Detect available device for PyTorch computations with robust fallback.

        Returns:
            torch.device to use
        """
        try:
            # Check CUDA with runtime validation
            if torch.cuda.is_available():
                try:
                    # Test GPU tensor creation to verify runtime availability
                    test_tensor = torch.tensor(
                        [GPU_TEST_VALUE], device="cuda:0"
                    )
                    del test_tensor  # Immediate cleanup
                    torch.cuda.empty_cache()

                    device = torch.device("cuda:0")
                    # Memory monitoring for diagnostics
                    if torch.cuda.is_available():
                        memory_allocated = torch.cuda.memory_allocated(device)
                        memory_reserved = torch.cuda.memory_reserved(device)
                        logging.info(
                            f"{self} - CUDA available, using GPU. "
                            f"Memory: allocated={memory_allocated//MEMORY_CONVERSION_FACTOR}MB, "
                            f"reserved={memory_reserved//MEMORY_CONVERSION_FACTOR}MB"
                        )
                    return device
                except RuntimeError as cuda_error:
                    logging.warning(
                        f"{self} - CUDA runtime error during validation: {cuda_error}, "
                        f"falling back to next available device"
                    )

            # Check MPS (Apple Silicon) with runtime validation
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                try:
                    # Test MPS tensor creation to verify runtime availability
                    test_tensor = torch.tensor(
                        [GPU_TEST_VALUE], device="mps"
                    )
                    del test_tensor  # Immediate cleanup

                    device = torch.device("mps")
                    logging.info(f"{self} - MPS available, using Apple Silicon GPU")
                    return device
                except RuntimeError as mps_error:
                    logging.warning(
                        f"{self} - MPS runtime error during validation: {mps_error}, "
                        f"falling back to CPU"
                    )

            # Fallback to CPU
            device = torch.device("cpu")
            logging.info(f"{self} - Using CPU for N-BEATS computations")
            return device

        except Exception as e:
            logging.warning(f"{self} - Device detection error: {e}, using CPU fallback")
            return torch.device("cpu")

    def _create_model(self) -> Any:
        """
        Create N-BEATS model based on configuration.

        Returns:
            Initialized PyTorch model
        """

        class NBEATSModel(nn.Module):
            """Interpretable N-BEATS model with trend/seasonal stacks."""

            def __init__(self, config, device):
                super().__init__()
                self.config = config
                self.device = device
                self.stacks = nn.ModuleList()

                # Create stacks by type
                for stack_type in config["stack_types"]:
                    stack = self._create_stack(stack_type, config)
                    self.stacks.append(stack)

            def _create_stack(self, stack_type: str, config: Dict) -> nn.ModuleList:
                """Create stack of blocks of specific type."""
                stack = nn.ModuleList()

                for _ in range(config["num_blocks"]):
                    if stack_type == "trend":
                        block = self._create_trend_block(config)
                    elif stack_type == "seasonality":
                        block = self._create_seasonal_block(config)
                    else:  # generic
                        block = self._create_generic_block(config)
                    stack.append(block)

                return stack

            def _create_trend_block(self, config: Dict) -> nn.Module:
                """Create interpretable trend block with polynomial basis."""

                class TrendBlock(nn.Module):
                    def __init__(
                        self, input_size, forecast_size, num_layers, layer_width, degree
                    ):
                        super().__init__()
                        self.input_size = input_size
                        self.forecast_size = forecast_size
                        self.degree = degree

                        # Fully connected layers
                        layers = [nn.Linear(input_size, layer_width), nn.ReLU()]
                        for _ in range(num_layers - 1):
                            layers.extend(
                                [nn.Linear(layer_width, layer_width), nn.ReLU()]
                            )
                        self.fc_layers = nn.Sequential(*layers)

                        # Polynomial coefficient output layers
                        self.theta_b = nn.Linear(layer_width, degree + 1)
                        self.theta_f = nn.Linear(layer_width, degree + 1)

                    def forward(self, x):
                        hidden = self.fc_layers(x)
                        theta_b = self.theta_b(hidden)
                        theta_f = self.theta_f(hidden)

                        # Polynomial basis functions
                        t_b = torch.linspace(0, 1, self.input_size, device=x.device)
                        t_f = torch.linspace(
                            1,
                            1 + self.forecast_size / self.input_size,
                            self.forecast_size,
                            device=x.device,
                        )

                        basis_b = torch.stack(
                            [t_b**i for i in range(self.degree + 1)], dim=1
                        )
                        basis_f = torch.stack(
                            [t_f**i for i in range(self.degree + 1)], dim=1
                        )

                        backcast = torch.matmul(theta_b, basis_b.T)
                        forecast = torch.matmul(theta_f, basis_f.T)

                        return backcast, forecast

                return TrendBlock(
                    config["input_size"],
                    config["forecast_size"],
                    config["num_layers"],
                    config["layer_width"],
                    config["polynomial_degree"],
                )

            def _create_seasonal_block(self, config: Dict) -> nn.Module:
                """Create interpretable seasonal block with trigonometric basis."""

                class SeasonalBlock(nn.Module):
                    def __init__(
                        self,
                        input_size,
                        forecast_size,
                        num_layers,
                        layer_width,
                        harmonics,
                    ):
                        super().__init__()
                        self.input_size = input_size
                        self.forecast_size = forecast_size
                        self.harmonics = harmonics

                        # Fully connected layers
                        layers = [nn.Linear(input_size, layer_width), nn.ReLU()]
                        for _ in range(num_layers - 1):
                            layers.extend(
                                [nn.Linear(layer_width, layer_width), nn.ReLU()]
                            )
                        self.fc_layers = nn.Sequential(*layers)

                        # Fourier coefficient output layers
                        self.theta_b = nn.Linear(layer_width, 2 * harmonics)
                        self.theta_f = nn.Linear(layer_width, 2 * harmonics)

                    def forward(self, x):
                        hidden = self.fc_layers(x)
                        theta_b = self.theta_b(hidden)
                        theta_f = self.theta_f(hidden)

                        # Fourier basis functions
                        t_b = torch.linspace(0, 1, self.input_size, device=x.device)
                        t_f = torch.linspace(
                            1,
                            1 + self.forecast_size / self.input_size,
                            self.forecast_size,
                            device=x.device,
                        )

                        basis_b = []
                        basis_f = []
                        for i in range(1, self.harmonics + 1):
                            basis_b.extend(
                                [
                                    torch.cos(2 * np.pi * i * t_b),
                                    torch.sin(2 * np.pi * i * t_b),
                                ]
                            )
                            basis_f.extend(
                                [
                                    torch.cos(2 * np.pi * i * t_f),
                                    torch.sin(2 * np.pi * i * t_f),
                                ]
                            )

                        basis_b = torch.stack(basis_b, dim=1)
                        basis_f = torch.stack(basis_f, dim=1)

                        backcast = torch.matmul(theta_b, basis_b.T)
                        forecast = torch.matmul(theta_f, basis_f.T)

                        return backcast, forecast

                return SeasonalBlock(
                    config["input_size"],
                    config["forecast_size"],
                    config["num_layers"],
                    config["layer_width"],
                    config["harmonics"],
                )

            def _create_generic_block(self, config: Dict) -> nn.Module:
                """Create generic block for complex patterns."""

                class GenericBlock(nn.Module):
                    def __init__(
                        self, input_size, forecast_size, num_layers, layer_width
                    ):
                        super().__init__()
                        self.input_size = input_size
                        self.forecast_size = forecast_size

                        # Fully connected layers
                        layers = [nn.Linear(input_size, layer_width), nn.ReLU()]
                        for _ in range(num_layers - 1):
                            layers.extend(
                                [nn.Linear(layer_width, layer_width), nn.ReLU()]
                            )
                        self.fc_layers = nn.Sequential(*layers)

                        # Output layers
                        self.backcast_fc = nn.Linear(layer_width, input_size)
                        self.forecast_fc = nn.Linear(layer_width, forecast_size)

                    def forward(self, x):
                        hidden = self.fc_layers(x)
                        backcast = self.backcast_fc(hidden)
                        forecast = self.forecast_fc(hidden)
                        return backcast, forecast

                return GenericBlock(
                    config["input_size"],
                    config["forecast_size"],
                    config["num_layers"],
                    config["layer_width"],
                )

            def forward(self, x):
                """Forward pass through all stacks with residual connections."""
                residual = x
                forecast = torch.zeros(
                    x.size(0), self.config["forecast_size"], device=x.device
                )
                stack_outputs = {}

                for stack_idx, (stack_type, stack) in enumerate(
                    zip(self.config["stack_types"], self.stacks)
                ):
                    stack_backcast = torch.zeros_like(residual)
                    stack_forecast = torch.zeros(
                        x.size(0), self.config["forecast_size"], device=x.device
                    )

                    for block in stack:
                        backcast, block_forecast = block(residual)
                        stack_backcast += backcast
                        stack_forecast += block_forecast
                        residual = residual - backcast  # Residual connection

                    forecast += stack_forecast
                    stack_outputs[stack_type] = {
                        "backcast": stack_backcast,
                        "forecast": stack_forecast,
                    }

                return forecast, stack_outputs

        return NBEATSModel(self.config, self.device).to(self.device)

    def _prepare_training_data(self, data: pd.Series) -> Tuple[Any, Any, float, float]:
        """
        Prepare data for N-BEATS model training.

        Args:
            data: Time series

        Returns:
            Tuple[input sequences, target values, mean, std]
        """
        values = data.values.astype(FLOAT32_DTYPE)
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))

        # Protection against division by zero
        if std_val == 0:
            std_val = STD_ZERO_FALLBACK

        # Data normalization
        normalized_values = (values - mean_val) / std_val

        # Create training sequences
        input_size = self.config["input_size"]
        forecast_size = self.config["forecast_size"]

        sequences = []
        targets = []

        for i in range(len(normalized_values) - input_size - forecast_size + 1):
            seq = normalized_values[i : i + input_size]
            target = normalized_values[i + input_size : i + input_size + forecast_size]
            sequences.append(seq)
            targets.append(target)

        if not sequences:
            raise ValueError("Insufficient data for creating training sequences")

        X = torch.FloatTensor(np.array(sequences))
        y = torch.FloatTensor(np.array(targets))

        return X, y, mean_val, std_val

    def _train_and_decompose(
        self, data: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series, Dict[str, Any]]:
        """
        Train N-BEATS model and perform decomposition with proper GPU memory management.

        Args:
            data: Time series

        Returns:
            Tuple[trend, seasonality, residuals, model info]
        """
        model = None
        X, y = None, None
        initial_memory_allocated = 0
        initial_memory_reserved = 0

        try:
            # Memory monitoring - initial state
            if self.device.type == "cuda" and torch.cuda.is_available():
                initial_memory_allocated = torch.cuda.memory_allocated(self.device)
                initial_memory_reserved = torch.cuda.memory_reserved(self.device)
                logging.debug(
                    f"{self} - Initial GPU memory: "
                    f"allocated={initial_memory_allocated//MEMORY_CONVERSION_FACTOR}MB, "
                    f"reserved={initial_memory_reserved//MEMORY_CONVERSION_FACTOR}MB"
                )

            # Prepare training data
            X, y, mean_val, std_val = self._prepare_training_data(data)

            if len(X) == 0:
                # For compatibility with return type
                x = np.arange(len(data))
                coeffs = np.polyfit(x, data.values.astype(float), 1)
                trend_values = np.polyval(coeffs, x)
                trend = pd.Series(trend_values, index=data.index)
                seasonal = pd.Series(np.zeros(len(data)), index=data.index)
                residual = data - trend
                model_info = {
                    "final_loss": float("inf"),
                    "fallback_used": True,
                    "fallback_reason": "no_training_sequences",
                }
                return trend, seasonal, residual, model_info

            # Create and initialize model with GPU error handling
            try:
                model = self._create_model()

                # Transfer data to device with runtime GPU validation
                X = X.to(self.device)
                y = y.to(self.device)

                # Test simple GPU operation to verify availability
                if self.device.type in ["cuda", "mps"]:
                    test_tensor = torch.ones(1, device=self.device)
                    _ = test_tensor + test_tensor  # Simple operation test
                    del test_tensor

            except (RuntimeError, torch.cuda.OutOfMemoryError) as gpu_error:
                logging.warning(
                    f"{self} - GPU error during model/data setup: {gpu_error}. "
                    f"Falling back to CPU execution."
                )
                # Graceful fallback to CPU
                self.device = torch.device("cpu")
                if model is not None:
                    model = model.to(self.device)
                X = X.to(self.device) if X is not None else X
                y = y.to(self.device) if y is not None else y

                logging.info(f"{self} - Successfully migrated to CPU after GPU error")

            # Memory monitoring after loading data to GPU
            if self.device.type == "cuda" and torch.cuda.is_available():
                data_memory_allocated = torch.cuda.memory_allocated(self.device)
                logging.debug(
                    f"{self} - GPU memory after data transfer: "
                    f"allocated={data_memory_allocated//MEMORY_CONVERSION_FACTOR}MB "
                    f"(+{(data_memory_allocated - initial_memory_allocated)//MEMORY_CONVERSION_FACTOR}MB)"
                )

            # Prepare for training
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(
                dataset, batch_size=self.config["batch_size"], shuffle=True
            )

            optimizer = optim.Adam(model.parameters(), lr=self.config["learning_rate"])
            criterion = nn.MSELoss()

            # Train model with periodic memory monitoring and GPU error handling
            model.train()
            final_loss = INITIAL_LOSS
            gpu_fallback_triggered = False

            # Enhanced early stopping tracking
            best_loss = float("inf")
            patience_counter = 0
            loss_history = []

            for epoch in range(self.config["epochs"]):
                epoch_loss = INITIAL_LOSS
                batch_count = BATCH_COUNT_INITIAL

                for batch_x, batch_y in dataloader:
                    try:
                        optimizer.zero_grad()

                        forecast, _ = model(batch_x)
                        loss = criterion(forecast, batch_y)

                        loss.backward()

                        # Gradient clipping if configured
                        if (
                            "gradient_clipping" in self.config
                            and self.config["gradient_clipping"]
                        ):
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), self.config["gradient_clipping"]
                            )

                        optimizer.step()

                        epoch_loss += loss.item()
                        batch_count += 1

                    except (
                        RuntimeError,
                        torch.cuda.OutOfMemoryError,
                    ) as training_error:
                        if not gpu_fallback_triggered and self.device.type in [
                            "cuda",
                            "mps",
                        ]:
                            logging.warning(
                                f"{self} - GPU error during training at epoch {epoch}: {training_error}. "
                                f"Attempting CPU fallback."
                            )

                            # Attempt fallback to CPU (once per training)
                            gpu_fallback_triggered = True
                            try:
                                # Release GPU memory
                                if self.device.type == "cuda":
                                    torch.cuda.empty_cache()

                                # Migrate to CPU
                                self.device = torch.device("cpu")
                                model = model.to(self.device)

                                # Transfer current batch to CPU
                                batch_x = batch_x.to(self.device)
                                batch_y = batch_y.to(self.device)

                                # Continue training on CPU
                                optimizer = optim.Adam(
                                    model.parameters(), lr=self.config["learning_rate"]
                                )

                                # Repeat operations on CPU
                                optimizer.zero_grad()
                                forecast, _ = model(batch_x)
                                loss = criterion(forecast, batch_y)
                                loss.backward()
                                optimizer.step()

                                epoch_loss += loss.item()
                                batch_count += 1

                                logging.info(
                                    f"{self} - Successfully continued training on CPU"
                                )

                            except Exception as cpu_fallback_error:
                                logging.error(
                                    f"{self} - CPU fallback failed: {cpu_fallback_error}"
                                )
                                # Skip this batch but continue
                                continue
                        else:
                            # If fallback was already attempted or not a GPU error, skip batch
                            logging.warning(
                                f"{self} - Skipping batch due to training error: {training_error}"
                            )
                            continue

                # Protection against division by zero if all batches failed
                final_loss = epoch_loss / max(batch_count, 1)

                # Enhanced early stopping logic
                loss_history.append(final_loss)

                # Check for improvement
                if (
                    final_loss
                    < best_loss - EARLY_STOPPING_MIN_IMPROVEMENT
                ):
                    best_loss = final_loss
                    patience_counter = 0
                    logging.debug(
                        f"{self} - New best loss: {best_loss:.6f} at epoch {epoch}"
                    )
                else:
                    patience_counter += 1

                # Enhanced early stopping with patience and stability checks
                should_stop_early = False

                # Patience-based stopping
                if patience_counter >= EARLY_STOPPING_PATIENCE_EPOCHS:
                    logging.info(
                        f"{self} - Enhanced early stopping: no improvement for {patience_counter} epochs "
                        f"(best loss: {best_loss:.6f})"
                    )
                    should_stop_early = True

                # Loss stability check (convergence)
                if (
                    len(loss_history) >= CONVERGENCE_STABILITY_EPOCHS
                    and len(
                        set(
                            f"{x:.6f}"
                            for x in loss_history[
                                -CONVERGENCE_STABILITY_EPOCHS :
                            ]
                        )
                    )
                    == 1
                ):
                    logging.info(
                        f"{self} - Enhanced early stopping: loss stabilized at {final_loss:.6f} for "
                        f"{CONVERGENCE_STABILITY_EPOCHS} epochs"
                    )
                    should_stop_early = True

                if should_stop_early:
                    break

                # Log progress with memory monitoring
                if (
                    epoch
                    % max(
                        MIN_PROGRESS_LOG_INTERVAL,
                        self.config["epochs"] // PROGRESS_LOG_DIVISOR,
                    )
                    == 0
                ):
                    current_memory = 0
                    if self.device.type == "cuda" and torch.cuda.is_available():
                        current_memory = torch.cuda.memory_allocated(self.device)

                    logging.debug(
                        f"{self} - Epoch {epoch}/{self.config['epochs']}, "
                        f"Loss: {final_loss:.6f}, "
                        f"GPU Memory: {current_memory//MEMORY_CONVERSION_FACTOR}MB"
                    )

                # Early stopping if configured
                if (
                    "early_stopping_patience" in self.config
                    and self.config["early_stopping_patience"]
                    and final_loss < EARLY_STOPPING_LOSS_THRESHOLD
                ):
                    logging.info(
                        f"{self} - Early stopping at epoch {epoch} (loss: {final_loss:.6f})"
                    )
                    break

            # Extract decomposition components
            trend, seasonal, residual = self._extract_components(
                model, data, mean_val, std_val
            )

            # Memory monitoring after training
            peak_memory_allocated = 0
            peak_memory_reserved = 0
            if self.device.type == "cuda" and torch.cuda.is_available():
                peak_memory_allocated = torch.cuda.memory_allocated(self.device)
                peak_memory_reserved = torch.cuda.memory_reserved(self.device)

            # Model info with memory usage
            model_info = {
                "final_loss": final_loss,
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "training_sequences": len(X),
                "converged": final_loss
                < CONVERGENCE_LOSS_THRESHOLD,  # Simple convergence check
                "peak_memory_mb": (
                    peak_memory_allocated // 1024 // 1024
                    if peak_memory_allocated > 0
                    else 0
                ),
                "memory_growth_mb": (
                    (peak_memory_allocated - initial_memory_allocated) // 1024 // 1024
                    if peak_memory_allocated > 0
                    else 0
                ),
            }

            logging.info(
                f"{self} - N-BEATS training completed: "
                f"loss={final_loss:.6f}, "
                f"parameters={model_info['total_parameters']}, "
                f"peak_memory={model_info['peak_memory_mb']}MB"
            )

            return trend, seasonal, residual, model_info

        except Exception as e:
            logging.error(f"{self} - Training error: {str(e)}")
            # For compatibility with return type, create simple components
            x = np.arange(len(data))
            coeffs = np.polyfit(x, data.values.astype(float), 1)
            trend_values = np.polyval(coeffs, x)
            trend = pd.Series(trend_values, index=data.index)
            seasonal = pd.Series(np.zeros(len(data)), index=data.index)
            residual = data - trend
            model_info = {
                "final_loss": float("inf"),
                "fallback_used": True,
                "training_error": str(e),
            }
            return trend, seasonal, residual, model_info

        finally:
            # Critical GPU cleanup to prevent memory leaks
            try:
                if model is not None:
                    # Set model to eval mode and clean up
                    model.eval()
                    # Delete model from memory
                    del model

                # Clean up data tensors
                if X is not None:
                    del X
                if y is not None:
                    del y

                # Force GPU cache cleanup
                if self.device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                    # Final memory monitoring
                    final_memory_allocated = torch.cuda.memory_allocated(self.device)
                    final_memory_reserved = torch.cuda.memory_reserved(self.device)

                    logging.debug(
                        f"{self} - GPU cleanup completed. "
                        f"Final memory: allocated={final_memory_allocated//MEMORY_CONVERSION_FACTOR}MB, "
                        f"reserved={final_memory_reserved//MEMORY_CONVERSION_FACTOR}MB, "
                        f"released={(initial_memory_allocated - final_memory_allocated)//MEMORY_CONVERSION_FACTOR}MB"
                    )
                elif self.device.type == "mps":
                    # MPS cleanup (if supported)
                    if hasattr(torch.mps, "empty_cache"):
                        torch.mps.empty_cache()  # type: ignore
                    logging.debug(f"{self} - MPS memory cleanup completed")

            except Exception as cleanup_error:
                logging.warning(
                    f"{self} - GPU cleanup error (non-critical): {cleanup_error}"
                )

    def _extract_components(
        self,
        model: Any,
        original_data: pd.Series,
        mean_val: float,
        std_val: float,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Extract decomposition components from trained N-BEATS model.

        Args:
            model: Trained model
            original_data: Original data
            mean_val: Mean for denormalization
            std_val: Standard deviation for denormalization

        Returns:
            Tuple[trend, seasonality, residuals]
        """
        model.eval()

        n = len(original_data)
        input_size = self.config["input_size"]

        # Initialize components
        trend_component = np.zeros(n)
        seasonal_component = np.zeros(n)
        overlap_counts = np.zeros(n)

        # Normalized data
        normalized_data = (original_data.values.astype(float) - mean_val) / std_val

        with torch.no_grad():
            # Sliding window over all possible positions
            for i in range(n - input_size + 1):
                window = normalized_data[i : i + input_size]
                x = torch.FloatTensor(window).unsqueeze(0).to(self.device)

                _, stack_outputs = model(x)

                # Extract backcast for each stack type
                for stack_type, output in stack_outputs.items():
                    backcast = output["backcast"].cpu().numpy().flatten()

                    # Denormalization
                    backcast = backcast * std_val + mean_val

                    if stack_type == "trend":
                        trend_component[i : i + input_size] += backcast
                    elif stack_type == "seasonality":
                        seasonal_component[i : i + input_size] += backcast
                    else:  # generic - split equally
                        trend_component[i : i + input_size] += (
                            backcast * GENERIC_SPLIT_RATIO
                        )
                        seasonal_component[i : i + input_size] += (
                            backcast * GENERIC_SPLIT_RATIO
                        )

                overlap_counts[i : i + input_size] += 1

            # Normalize by overlap count
            overlap_counts = np.maximum(overlap_counts, 1)
            trend_component /= overlap_counts
            seasonal_component /= overlap_counts

            # Handle boundary values
            self._handle_boundaries(trend_component, seasonal_component, input_size)

        # Create pandas Series
        trend = pd.Series(trend_component, index=original_data.index)
        seasonal = pd.Series(seasonal_component, index=original_data.index)
        residual = original_data - trend - seasonal

        return trend, seasonal, residual

    def _handle_boundaries(
        self,
        trend_component: np.ndarray,
        seasonal_component: np.ndarray,
        input_size: int,
    ) -> None:
        """
        Handle boundary values of components for smooth transitions.
        Vectorized version for improved performance.

        Args:
            trend_component: Trend component array
            seasonal_component: Seasonal component array
            input_size: Input window size
        """
        if len(trend_component) > input_size:
            boundary_size = min(input_size // 2, 5)  # Limited boundary

            # Vectorized fill of series start
            start_fill_value_trend = trend_component[boundary_size]
            start_fill_value_seasonal = seasonal_component[boundary_size]

            trend_component[:boundary_size] = start_fill_value_trend
            seasonal_component[:boundary_size] = start_fill_value_seasonal

            # Vectorized fill of series end
            end_fill_value_trend = trend_component[-(boundary_size + 1)]
            end_fill_value_seasonal = seasonal_component[-(boundary_size + 1)]

            trend_component[-boundary_size:] = end_fill_value_trend
            seasonal_component[-boundary_size:] = end_fill_value_seasonal

    def _fallback_decomposition(
        self, data: pd.Series, context_params: Dict[str, Any], reason: str
    ) -> Dict[str, Any]:
        """
        Simple linear decomposition as fallback when N-BEATS is not possible.

        Args:
            data: Time series
            context_params: Context parameters
            reason: Reason for using fallback

        Returns:
            Tuple[trend, seasonality, residuals, model info]
        """
        logging.warning(f"{self} - Using fallback decomposition: {reason}")

        # Simple linear trend through polynomial regression
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data.values.astype(float), 1)
        trend_values = np.polyval(coeffs, x)
        trend = pd.Series(trend_values, index=data.index)

        # Zero seasonality (N-BEATS cannot train)
        seasonal = pd.Series(np.zeros(len(data)), index=data.index)

        # Residuals
        residual = data - trend

        # Prepare additional data
        additional_data = {
            "model_type": "linear_fallback",
            "fallback_used": True,
            "fallback_reason": reason,
            "training_loss": float("inf"),
            "device_used": str(self.device),
        }

        # Use base method to prepare result
        return self.prepare_decomposition_result(
            trend, seasonal, residual, data, context_params, additional_data
        )