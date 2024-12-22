# type: ignore

import pandas as pd
from .BaseDailyDataProcessor import BaseDailyDataProcessor
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler

class CalculateDailyPeriodicActivityLevels(BaseDailyDataProcessor):
    def __init__(self, all_days_df: pd.DataFrame):
        super().__init__()
        self.all_days_df = all_days_df

    def process(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        data = {
            "day": [],
            "daily_periodic_activity_levels": [],
        }

        all_days_df = self.all_days_df

        for day in self.all_days_df.index:
            data["day"].append(day)

            daily_periodic_df: pd.DataFrame = all_days_df[all_days_df.index == day]['daily_periodic_df'].iloc[0].copy() # type: ignore

            daily_periodic_df = daily_periodic_df.reset_index()
            daily_periodic_df.drop(columns=['period'], inplace=True)

            scaler = StandardScaler()
            standardized_data = scaler.fit_transform(daily_periodic_df)

            pca = PCA()
            pca.fit_transform(standardized_data)
            pc1_weight = pca.components_[0]

            periodic_activities = np.dot(standardized_data, pc1_weight)
            periodic_activities_series = pd.Series(periodic_activities, index=self.period_range())
            data["daily_periodic_activity_levels"].append(periodic_activities_series)

        df = pd.DataFrame(data)
        df.set_index('day', inplace=True)
        return all_days_df, df
