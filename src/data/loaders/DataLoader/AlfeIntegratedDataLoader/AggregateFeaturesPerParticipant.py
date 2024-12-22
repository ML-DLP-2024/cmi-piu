from typing import Any
from .BaseDailyDataProcessor import BaseDailyDataProcessor
from .EngineerFeaturesFromPeriodicActivityLevels import EngineerFeaturesFromPeriodicActivityLevels
import pandas as pd

class AggregateFeaturesPerParticipant(BaseDailyDataProcessor):
    def __init__(self, efdf: pd.DataFrame):
        self.efdf = efdf.copy()

    AGGREGATE_NAMES = ['mean', 'min', 'max', 'std']
    
    def process(self):
        df = self.efdf

        features: list[str] = []
        values: list[Any] = []

        for col in df.columns:
            aggregates: pd.Series[Any] = df[col].agg(self.AGGREGATE_NAMES) # type: ignore
            for aggregate_name in self.AGGREGATE_NAMES:
                features.append( f"{col}_{aggregate_name.upper()}" )
                v: Any = aggregates[aggregate_name]
                values.append(v)

        return pd.Series(values, index=features)

    @classmethod
    def get_features(cls):
        features: list[str] = []
        for col in EngineerFeaturesFromPeriodicActivityLevels.get_features():
            for aggregate_name in cls.AGGREGATE_NAMES:
                features.append( f"{col}_{aggregate_name.upper()}" )
        return features
