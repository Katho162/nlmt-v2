# NLMT-v2 Language Recommender

## Project Overview

This project is a language recommender system that suggests languages for users to learn based on the languages they already know. It is built with Python and uses a Variational Autoencoder (VAE) model implemented with PyTorch.

The project provides three main functionalities:

1.  **Training:** A script to train the VAE model on a dataset of user language skills.
2.  **Recommendation:** A command-line interface (CLI) and a FastAPI endpoint to get language recommendations.
3.  **Hyperparameter Tuning:** A script to automatically tune the hyperparameters of the model using Optuna.

## Building and Running

### Installation

To install the project and its dependencies, run the following command:

```bash
pip install -e .
```

### Training

To train the model with default hyperparameters, run the `train.py` script:

```bash
python scripts/train.py
```

This will train the VAE model and save the trained model to the `models/` directory.

To train the model with a specific set of hyperparameters (e.g., from hyperparameter tuning), provide a path to a hyperparameter configuration file:

```bash
python scripts/train.py --hyperparameters best_hyperparameters.json
```

### Recommendation

#### CLI

To get language recommendations from the command line, use the `recommend.py` script:

```bash
python scripts/recommend.py English Japanese
```

#### API

To run the FastAPI application, use the following command:

```bash
uvicorn nlmt_v2.api.main:app --reload --app-dir src
```

This will start a local server. You can then send a POST request to `http://127.0.0.1:8000/recommend`:

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "known_languages": ["English", "Japanese"],
  "top_k": 5
}' http://127.0.0.1:8000/recommend
```

### Hyperparameter Tuning

To automatically tune the hyperparameters of the model, you can use the `tune_hyperparameters.py` script:

```bash
python scripts/tune_hyperparameters.py --n-trials 100
```

This will run 100 trials and save the best hyperparameters to a `best_hyperparameters.json` file. You can then use this file to train the model as described in the **Training** section.

## Development Conventions

- **Project Structure:** The project follows a standard Python project structure, with source code in the `src/` directory and scripts in the `scripts/` directory.
- **Configuration:** Hyperparameters are managed using Pydantic models in the `src/nlmt_v2/config` directory.
- **Model Versioning:** Trained models are saved with a timestamp to allow for versioning.
