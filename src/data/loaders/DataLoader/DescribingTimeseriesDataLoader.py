import os
import pandas as pd
from .BaseDataLoader import BaseDataLoader
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import override
from src.utils.context import DataProcessingContext

class DescribingTimeseriesDataLoader(BaseDataLoader):
    @override
    def do_load(self, dataset: str) -> pd.DataFrame:
        def process_file(filename: str, dirname: str): # type: ignore
            df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
            df.drop('step', axis=1, inplace=True)
            return df.describe().values.reshape(-1), filename.split('=')[1] # type: ignore

        def load_time_series_internal(dirname: str) -> pd.DataFrame:
            ids = os.listdir(dirname)
            
            with ThreadPoolExecutor() as executor:
                results = list(tqdm(executor.map(lambda fname: process_file(fname, dirname), ids), total=len(ids))) # type: ignore
            
            stats, indexes = zip(*results)
            
            df = pd.DataFrame(stats, columns=[f"stat_{i}" for i in range(len(stats[0]))])
            DataProcessingContext.get_instance()["time_series_cols"] = df.columns.to_list()
            df['id'] = indexes
            return df
        
        return load_time_series_internal(os.path.join(self.data_dir, f"series_{dataset}.parquet"))
