from src.configs.implementation_configs import DatasetConfig
from src.datasets.musdb18_dataset import MusDBDataset
from src.datasets.musdb18_stem_dataset import MusDBStemDataset


class DatasetFactory:
    @staticmethod
    def build(config: DatasetConfig):
        match config.name:
            case "MusDBStemDataset":
                dataset = MusDBStemDataset(**config.kwargs)
            case "MusDBDataset":
                dataset = MusDBDataset(**config.kwargs)
            case _:
                raise ValueError(f"Not supported dataset {config.name}")
        return dataset
