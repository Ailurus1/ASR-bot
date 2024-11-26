from abc import ABC, abstractmethod
import polars as pl
from datasets import load_dataset
from typing import Dict, Any


class PreparedDataset(ABC):
    def __init__(self, name: str, dataset_kwargs: Dict[str, Any] = {}):
        self.name = name
        self.dataset_kwargs = dataset_kwargs
        self.dataset = None

    def load_dataset(self) -> None:
        try:
            self.dataset = load_dataset(self.name, **self.dataset_kwargs)
        except Exception as exc:
            raise exc

    @abstractmethod
    def get_eval_dataset(self, samplint_rate: int) -> pl.DataFrame:
        raise NotImplementedError("Subclasses should implement this method")
