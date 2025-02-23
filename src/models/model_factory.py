from torch.nn import Module

from src.configs.implementation_configs import ModelConfig
from src.models.custom_music2latent import CustomMusic2Latent


class ModelFactory:
    @staticmethod
    def build(config: ModelConfig) -> Module:
        match config.name:
            case "CustomMusic2Latent":
                model = CustomMusic2Latent(**config.kwargs)
            case _:
                raise ValueError(f"Not supported dataset {config.name}")
        return model
