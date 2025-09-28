import torch
import torch.nn as nn


class VAERecommender(nn.Module):
    def __init__(self, num_languages, latent_dim=64):
        super().__init__()
        self.num_languages = num_languages
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_languages, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        # Mean and log-variance for latent space
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_languages),
        )

        # Language embeddings (optional)
        self.language_embeddings = nn.Embedding(num_languages, latent_dim)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # Optionally add mean of language embeddings for known languages
        lang_idx = x.nonzero(as_tuple=False)[:, 1]  # indices of known languages
        if len(lang_idx) > 0:
            z += self.language_embeddings(lang_idx).mean(dim=0)
        out = self.decode(z)
        return out, mu, logvar
