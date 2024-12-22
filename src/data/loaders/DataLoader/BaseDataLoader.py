import pandas as pd

class BaseDataLoader:
    def __init__(self, name: str, data_dir: str):
        self.name = name
        self.data_dir = data_dir
    
    def do_load(self, dataset: str) -> pd.DataFrame:
        raise NotImplementedError
    
    def load(self, dataset: str) -> pd.DataFrame:
        return self.do_load(dataset)
