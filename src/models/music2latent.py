import os
from loguru import logger
import torch
import torch.nn as nn
from music2latent.models import UNet
from music2latent.audio import wv2realimag, realimag2wv
from huggingface_hub import hf_hub_download


def download_model():
    folder = ".models"
    model_path = os.path.join(folder, "music2latent.pt")
    if not os.path.exists(model_path):
        logger.info("Downloading model...")
        os.makedirs(folder, exist_ok=True)
        model_path = hf_hub_download(
            repo_id="SonyCSLParis/music2latent", filename="music2latent.pt", cache_dir=folder, local_dir=folder
        )
        logger.info("Model was downloaded successfully!")
    return model_path


class Music2Latent2Source(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()

        self.pretrained = pretrained

        self.model = self.load_model()

    def load_model(self) -> UNet:
        model = UNet()
        if self.pretrained:
            model_path = download_model()
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint["gen_state_dict"], strict=False)
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            spec = wv2realimag(x)

        latent = self.model.encoder(spec)
        rec_spec = self.model.decoder(latent)

        with torch.no_grad():
            y = realimag2wv(rec_spec)

        return y
