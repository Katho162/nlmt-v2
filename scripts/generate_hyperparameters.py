import json
from nlmt_v2.config.hyperparameters import Hyperparameters

def generate_hyperparameters(output_path: str = "hyperparameters.json"):
    """Generates a hyperparameter configuration file."""
    hyperparameters = Hyperparameters()
    with open(output_path, "w") as f:
        json.dump(hyperparameters.dict(), f, indent=4)
    print(f"Hyperparameter configuration saved to {output_path}")

if __name__ == "__main__":
    generate_hyperparameters()
