from pydantic import BaseModel, Field

class Hyperparameters(BaseModel):
    latent_dim: int = Field(default=64, ge=16, le=256)
    learning_rate: float = Field(default=0.001, ge=1e-5, le=1e-2)
    epochs: int = Field(default=30, ge=1, le=100)
    batch_size: int = Field(default=256, ge=32, le=1024)
