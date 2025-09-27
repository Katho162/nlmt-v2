import torch
from torch.utils.data import DataLoader, TensorDataset
from nlmt_v2.data.data_loader import load_and_preprocess_data
from nlmt_v2.models.model import VAERecommender
from nlmt_v2.utils.loss import vae_loss
from nlmt_v2.config.hyperparameters import Hyperparameters
import pickle

DATA_PATH = "data/roles.csv"
MODEL_PATH = "models/vae_recommender.pth"
LANGUAGES_PATH = "models/languages.pkl"

def train(hyperparameters: Hyperparameters, is_tuning: bool = False):
    # Load and preprocess data
    merged_data = load_and_preprocess_data(DATA_PATH)
    X_tensor = torch.tensor(merged_data.values)
    dataset = TensorDataset(X_tensor, X_tensor)
    dataloader = DataLoader(dataset, batch_size=hyperparameters.batch_size, shuffle=True)

    # Initialize model and optimizer
    model = VAERecommender(X_tensor.shape[1], latent_dim=hyperparameters.latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters.learning_rate)
    epochs = hyperparameters.epochs

    # Training loop
    final_loss = 0
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            recon, mu, logvar = model(batch_x)
            loss = vae_loss(recon, batch_y, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        final_loss = total_loss / len(dataloader)
        if not is_tuning:
            print(f"Epoch {epoch+1}, Loss: {final_loss:.4f}")

    if not is_tuning:
        # Save the model and languages with a timestamp
        from datetime import datetime
        import os

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        MODEL_DIR = "models"
        model_path = os.path.join(MODEL_DIR, f"vae_recommender_{timestamp}.pth")
        languages_path = os.path.join(MODEL_DIR, f"languages_{timestamp}.pkl")

        torch.save(model.state_dict(), model_path)
        with open(languages_path, "wb") as f:
            pickle.dump(merged_data.columns.to_list(), f)

        print(f"Model saved to {model_path}")
        print(f"Languages saved to {languages_path}")

    return final_loss

if __name__ == "__main__":
    import argparse
    import json


    parser = argparse.ArgumentParser(description="Train the VAE recommender model.")
    parser.add_argument("--hyperparameters", help="Path to hyperparameter configuration file.")
    args = parser.parse_args()

    if args.hyperparameters:
        with open(args.hyperparameters, "r") as f:
            config = json.load(f)
            hyperparameters = Hyperparameters(**config)
    else:
        hyperparameters = Hyperparameters()

    train(hyperparameters)
