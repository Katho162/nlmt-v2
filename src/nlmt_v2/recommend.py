import torch
import numpy as np
import pickle
from nlmt_v2.models.model import VAERecommender

MODEL_PATH = "models/vae_recommender.pth"
LANGUAGES_PATH = "models/languages.pkl"

def recommend(known_languages: list[str], top_k: int = 10):
    import os
    import glob

    # Find the latest model and languages file
    MODEL_DIR = "models"
    list_of_models = glob.glob(os.path.join(MODEL_DIR, "vae_recommender_*.pth"))
    if not list_of_models:
        raise FileNotFoundError("No trained models found.")
    latest_model_path = max(list_of_models, key=os.path.getctime)
    
    timestamp = latest_model_path.split('_')[-1].split('.')[0]
    languages_path = os.path.join(MODEL_DIR, f"languages_{timestamp}.pkl")
    if not os.path.exists(languages_path):
        raise FileNotFoundError(f"Languages file not found for model {latest_model_path}")

    # Load the model and languages
    with open(languages_path, "rb") as f:
        languages = pickle.load(f)
    
    model = VAERecommender(len(languages), latent_dim=64)
    model.load_state_dict(torch.load(latest_model_path))
    model.eval()

    # Create user vector
    user_vector = np.zeros(len(languages))
    for lang in known_languages:
        if lang in languages:
            user_vector[languages.index(lang)] = 1

    # Get recommendations
    with torch.no_grad():
        input_tensor = torch.tensor(user_vector, dtype=torch.float32).unsqueeze(0)
        recon, _, _ = model(input_tensor)
        preds = torch.sigmoid(recon).squeeze().numpy()

    preds[user_vector > 0] = -1  # mask known languages
    recommended_indices = preds.argsort()[-top_k:][::-1]
    
    return [languages[i] for i in recommended_indices]
