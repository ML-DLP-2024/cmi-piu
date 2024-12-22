from typing import Any
import pandas as pd

class BaseTrainer:
    def __init__(self, name: str):
        self.name = name
    
    def train(self, train: pd.DataFrame, test: pd.DataFrame, model_class: Any) -> pd.DataFrame:
        """Train the model and returns the prediction from test_data, as the form of pd.DataFrame with two columns, id and sii"""
        raise NotImplementedError