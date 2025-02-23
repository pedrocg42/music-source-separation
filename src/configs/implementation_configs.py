from typing import Any

from pydantic import BaseModel, Field


class ImplementationConfig(BaseModel()):
    name: str = Field(description="Name of the implementation")
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Arguments to create the class with")


class DatasetConfig(ImplementationConfig):
    pass


class ModelConfig(ImplementationConfig):
    pass
