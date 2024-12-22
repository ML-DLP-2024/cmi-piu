import pandas as pd
from .BaseDataLoader import BaseDataLoader
from typing import override

class TabularDataLoader(BaseDataLoader):
    @override
    def do_load(self, dataset: str) -> pd.DataFrame:
        df = pd.read_csv(f"{self.data_dir}/{dataset}.csv") # type: ignore
        return df
