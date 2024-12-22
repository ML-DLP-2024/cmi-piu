from ..BaseDataLoader import BaseDataLoader
from typing import override, Literal

from .CalculateDailyPeriodicActivityRelatedValues import CalculateDailyPeriodicActivityRelatedValues
from .CalculateDailyPeriodicActivityLevels import CalculateDailyPeriodicActivityLevels
from .EngineerFeaturesFromPeriodicActivityLevels import EngineerFeaturesFromPeriodicActivityLevels
from .AggregateFeaturesPerParticipant import AggregateFeaturesPerParticipant

import pandas as pd
import numpy as np

import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from src.utils.context import DataProcessingContext

class AlfeIntegratedDataLoader(BaseDataLoader):
    def read_worn_data(self, participant_id: str, dataset: Literal['train', 'test']) -> pd.DataFrame:
        data = pd.read_parquet(f'{self.data_dir}/series_{dataset}.parquet/id={participant_id}/part-0.parquet')
        worn_data = data[data['non-wear_flag'] == 0]
        return worn_data

    def timeseries_feature_engineering(self, participant_id: str|None, dataset: Literal['train', 'test']):
        try:
            if not participant_id or pd.isna(participant_id) or pd.isnull(participant_id):
                raise ValueError("Invalid participant ID")
            data = self.read_worn_data(participant_id, dataset=dataset)
        except Exception as e:
            print(f"Error while engineering features from timeseries data: {e}")
            return pd.Series(dict(
                (feature, np.nan) for feature in AggregateFeaturesPerParticipant.get_features()
            ))
        
        data = CalculateDailyPeriodicActivityRelatedValues(data).process()
        data = CalculateDailyPeriodicActivityLevels(data).process()
        data = EngineerFeaturesFromPeriodicActivityLevels(data).process()
        data = AggregateFeaturesPerParticipant(data).process()
        data['id'] = participant_id
        return data # pd.Series
    
    @override
    def do_load(self, dataset: str) -> pd.DataFrame:
        if dataset != 'train' and dataset != 'test':
            raise ValueError(f"Invalid dataset: {dataset}. If there is a new dataset, please add it to the list of valid datasets here.")

        dirname = os.path.join(self.data_dir, f"series_{dataset}.parquet")
        participant_ids = list(map(lambda x: x.split('=')[1], os.listdir(dirname)))

        def process_participant(participant_id: str):
            return self.timeseries_feature_engineering(participant_id, dataset)
        
        with ThreadPoolExecutor() as executor:
            results = list(
                tqdm(executor.map(
                        process_participant,
                        participant_ids,
                    ),
                    total=len(participant_ids)
                )
            )
        
        df = pd.DataFrame(results)

        DataProcessingContext.get_instance()['alfe_columns'] = AggregateFeaturesPerParticipant.get_features()

        return df
