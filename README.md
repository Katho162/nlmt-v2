# NLMT-v2 Language Recommender

This project is a language recommender system that suggests languages for users to learn based on the languages they already know. It uses a Variational Autoencoder (VAE) model trained on a dataset of user language skills.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Katho162/nlmt-v2
   cd nlmt-v2
   ```

2. **Install the dependencies:**

   ```bash
   pip install -e .
   ```

## Usage

### Training

To train the model, run the `train.py` script:

```bash
python scripts/train.py
```

This will train the VAE model and save the trained model to the `models/` directory.

### Recommendation

To get language recommendations, run the `recommend.py` script directly from the file, followed by the languages you already know. For example:

```bash
python scripts/recommend.py English Japanese
```

This will output a list of recommended languages.

## Project Structure

- `data/`: Contains the dataset used for training.
- `models/`: Stores the trained model and language list.
- `notebooks/`: Contains the Jupyter notebook used for model development.
- `scripts/`: Contains the scripts for training and recommendation.
- `src/`: Contains the source code for the project.
  - `nlmt_v2/`: The main package.
    - `data/`: Data loading and preprocessing.
    - `models/`: The VAE model definition.
    - `utils/`: Utility functions, such as the loss function.

## Automated Hyperparameter Tuning

To automatically tune the hyperparameters of the model, you can use the `tune_hyperparameters.py` script. This script uses Optuna to search for the best hyperparameter combination.

```bash
python scripts/tune_hyperparameters.py --n-trials 100
```

This will run 100 trials and print the best hyperparameters found. The best hyperparameters will also be saved to a `best_hyperparameters.json` file, which you can then use to train the final model.

## API Usage

To run the FastAPI application, use the following command:

```bash
uvicorn nlmt_v2.api.main:app --reload --app-dir src
```

This will start a local server. You can then send a POST request to `http://127.0.0.1:8000/recommend` with a JSON body like this:

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "known_languages": ["English", "Japanese"],
  "top_k": 5
}' http://127.0.0.1:8000/recommend
```