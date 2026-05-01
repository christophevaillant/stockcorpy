from abc import ABC, abstractmethod
from pickle import dump, load
from pathlib import Path

class Model(ABC):

    def __init__(self, model_name):
        self.model_name = model_name

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def update_model(self):
        pass
        
    def save_model(self):
        filename = f"{self.model_name}.pickle"
        with open(filename, "wb") as model_file:
            dump(self, model_file)

    @staticmethod
    def load_from_file(filepath: Path):
        with open(filepath, "rb") as modelfile:
             return load(modelfile)

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def predict_next_day(self):
        pass