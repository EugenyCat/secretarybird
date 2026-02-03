import optuna
import logging


class HyperparameterOptimizer:
    def __init__(self, model, hyperparam_space, data, evaluator):
        """
        Initialization of the hyperparameter optimizer.

        Parameters:
        - model (BaseModel): Model for which hyperparameter optimization is performed.
        - hyperparam_space (dict): Hyperparameter space.
        - data (dict): Data required for model training.
        """
        self.model = model
        self.hyperparam_space = hyperparam_space
        self.data = data
        self.evaluator = evaluator
        self.best_params = None


    def objective(self, trial):
        """
        Objective function for Optuna, which will be called for each iteration.

        Parameters:
        - trial (optuna.trial.Trial): Optuna Trial instance used for optimization.

        Returns:
        - float: Loss function value (or metric that we minimize).
        """
        # Generate hyperparameters for the model based on the search space
        params = self._get_params_for_trial(trial)

        # Train the model with these hyperparameters
        self.model.train(self.data['X'], self.data['y'], **params)

        # Optuna selects one of the validation methods to use in the current run.
        validation_method = trial.suggest_categorical('validation_type', [
            self.evaluator.evaluate_using_time_series_split,
            self.evaluator.evaluate_using_rolling_window,
            self.evaluator.evaluate_using_expanding_window,
            self.evaluator.evaluate_using_walk_forward_validation,
            self.evaluator.evaluate_using_backtest
        ])

        # Evaluate model quality # todo: check the result
        score = validation_method(self.data['X'], self.data['y'])

        # Logging loss values
        logging.info(f"Trial: {trial.number}, Score: {score}")

        return score


    def _get_params_for_trial(self, trial):
        """
            Generate hyperparameters for each iteration based on the hyperparameter space.

            Parameters:
            - trial (optuna.trial.Trial): Optuna Trial instance.

            Returns:
            - dict: Dictionary with hyperparameters for this iteration.
        """
        params = {}
        for key in self.hyperparam_space:
            params[key] = trial.suggest_categorical(key, self.hyperparam_space[key])
        return params


    def evaluate_model(self, X_test, y_test, assessment="smape"):
        """
            Evaluate the model on validation data.

            Parameters:
            - X_test (tensor): Input data for validation.
            - y_test (tensor): Target values for validation.

            Returns:
            - float: Assessment value.
        """
        evaluation = self.evaluator.evaluate(X_test, y_test)[assessment]
        return evaluation


    def optimize(self, n_trials=50):
        """
            Run hyperparameter optimization using Optuna.

            Parameters:
            - n_trials (int): Number of trials (iterations) for optimization.
        """
        study = optuna.create_study(direction='minimize')  # Optimize for loss minimization
        study.optimize(self.objective, n_trials=n_trials)

        self.best_params = study.best_params
        logging.info(f"Best hyperparameters: {self.best_params}")
        return self.best_params


    def get_best_params(self):
        """
            Returns the best hyperparameters after optimization.

            Returns:
            - dict: Best parameters.
        """
        return self.best_params