import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import optuna
from nlmt_v2.config.hyperparameters import Hyperparameters
from scripts.train import train as train_model
import json

def objective(trial: optuna.Trial):
    # Sample hyperparameters
    hyperparameters = Hyperparameters(
        latent_dim=trial.suggest_int("latent_dim", 16, 256),
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        epochs=trial.suggest_int("epochs", 10, 50),
        batch_size=trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
    )

    # Train the model and get the final loss
    final_loss = train_model(hyperparameters, is_tuning=True)
    return final_loss

def tune_hyperparameters(n_trials: int = 100):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save the best hyperparameters
    with open("best_hyperparameters.json", "w") as f:
        json.dump(trial.params, f, indent=4)
    print("Best hyperparameters saved to best_hyperparameters.json")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tune hyperparameters for the VAE recommender model.")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of trials to run.")
    args = parser.parse_args()
    tune_hyperparameters(args.n_trials)
