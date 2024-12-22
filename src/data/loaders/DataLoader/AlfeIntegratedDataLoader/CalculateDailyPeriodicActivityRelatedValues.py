# type: ignore

from typing import Any, TypedDict
from .BaseDailyDataProcessor import BaseDailyDataProcessor
import pandas as pd
import numpy as np
from src.utils.SkillfulImputer import SkillfulImputer
from sklearn.impute import KNNImputer

class AllDaysDataTypedDict(TypedDict):
    day: list[int]
    daily_periodic_df: list[pd.DataFrame]

class CalculateDailyPeriodicActivityRelatedValues(BaseDailyDataProcessor):
    def __init__(self, worn_data: pd.DataFrame):
        super().__init__()
        self.worn_data = worn_data.copy()
        self.available_days = self.worn_data['relative_date_PCIAT'].unique()

    def _preprocess_periodically_aggregated_values(self) -> pd.DataFrame:
        def convert_daily_periodic_series_to_list(series: pd.Series) -> list[Any]:
            period_to_value_dict = {}
            for period in series.index:
                period_to_value_dict[period] = series[period]
            for period in self.period_range():
                period_to_value_dict[period] = period_to_value_dict.get(period, np.nan)
            values = list(period_to_value_dict.items())
            values.sort(key=lambda elem: elem[0])
            values = [ e[1] for e in values ]
            values = np.array([x if x is not None else np.nan for x in values])
            # a list of which the nth value is the actual value at the nth period of the day.
            # e.g. [4, 6, 2, 7, 5, 8] => 4 is the value at period 0, 8 is the value at period 5.
            return list(values)

        all_days_data = {
            "day": [],
            "daily_periodic_df": [],
        }
        
        for day in self.available_days:
            base_data = self.worn_data[self.worn_data['relative_date_PCIAT'] == day].copy()
            base_data['daily_period'] = base_data['time_of_day'] / 1e9 / (self.granularity_in_hours * 3600)
            periodic_group = base_data.groupby(
                base_data['daily_period'].astype(int)
            )

            periodic_enmo = periodic_group['enmo'].agg(['mean', 'max', 'std'])
            periodic_light = periodic_group['light'].agg(['mean', 'max'])

            periodic_enmo_mean = periodic_enmo['mean'] # pd.Series
            periodic_enmo_max = periodic_enmo['max'] # pd.Series
            periodic_enmo_std = periodic_enmo['std'] # pd.Series
            periodic_anglez_std = periodic_group['anglez'].agg(['std'])['std'] # pd.Series
            periodic_anglez_sum_of_changes = periodic_group['anglez'].apply(lambda x: np.sum(np.abs(np.diff(x)))) # pd.Series
            periodic_light_mean = periodic_light['mean']
            periodic_light_max = periodic_light['max']

            daily_periodic_data = {
                "period": np.array(list(self.period_range())),
                "periodic_enmo_mean": convert_daily_periodic_series_to_list(periodic_enmo_mean),
                "periodic_enmo_max": convert_daily_periodic_series_to_list(periodic_enmo_max),
                "periodic_enmo_std": convert_daily_periodic_series_to_list(periodic_enmo_std),
                "periodic_anglez_std": convert_daily_periodic_series_to_list(periodic_anglez_std),
                "periodic_anglez_sum_of_changes": convert_daily_periodic_series_to_list(periodic_anglez_sum_of_changes),
                "periodic_light_mean": convert_daily_periodic_series_to_list(periodic_light_mean),
                "periodic_light_max": convert_daily_periodic_series_to_list(periodic_light_max),
            }
            
            daily_periodic_df = pd.DataFrame(daily_periodic_data)
            # daily_periodic_df.set_index('period', inplace=True)
            
            all_days_data["day"].append(day)
            all_days_data["daily_periodic_df"].append(daily_periodic_df)

        all_days_df = pd.DataFrame(all_days_data)
        # all_days_df.set_index("day", inplace=True)
        return all_days_df

    def _impute_periodically_aggregated_values(self, all_days_df: pd.DataFrame) -> pd.DataFrame:
        if len(all_days_df.index) == 0:
            return all_days_df

        data_requiring_imputation = {
            "period": [],
        }
        reassembly_info = []
        daily_periodic_columns = set()

        for day in all_days_df['day']:
            daily_periodic_df = all_days_df[all_days_df['day'] == day]['daily_periodic_df'].iloc[0]
            for col in daily_periodic_df.columns:
                data_requiring_imputation[col] = data_requiring_imputation.get(col, [])
                daily_periodic_columns.add(col)
            for period in daily_periodic_df.index:
                values_at_period = daily_periodic_df[daily_periodic_df.index == period]
                for col in values_at_period.columns:
                    data_requiring_imputation[col].append(values_at_period[col])
                reassembly_info.append((day, period))

        df_requiring_imputation = pd.DataFrame(data_requiring_imputation)
        imputer = SkillfulImputer(KNNImputer(n_neighbors=2))
        imputed_df = imputer.fit_transform(df_requiring_imputation, categorical_columns=[])

        new_all_days_data = {
            "day": [],
            "daily_periodic_df": [],
        }

        # for _i, imputed_entry in imputed_df.iterrows():
        #     day, period = reassembly_info.pop(0)
        #     day = int(day)
        #     period = int(period)
        #     while True:
        #         try:
        #             i = new_all_days_data['day'].index(day)
        #         except ValueError: # day is not added yet
        #             print('check check 1')
        #             new_all_days_data['day'].append(day)
        #             new_all_days_data['daily_periodic_df'].append(
        #                 dict((col, []) for col in daily_periodic_columns)
        #             )
        #             continue
        #         else:
        #             daily_periodic_df = new_all_days_data['daily_periodic_df'][i]
        #             for col in daily_periodic_columns:
        #                 daily_periodic_df[col].append(imputed_entry[col])
        #             break

        for _i, imputed_entry in imputed_df.iterrows():
            day, period = reassembly_info.pop(0)
            day = int(day)
            period = int(period)
        
            # Kiểm tra xem 'day' đã có trong 'new_all_days_data['day']' hay chưa
            if day not in new_all_days_data['day']:
                new_all_days_data['day'].append(day)
                new_all_days_data['daily_periodic_df'].append(
                    dict((col, []) for col in daily_periodic_columns)
                )
        
            # Sau khi chắc chắn 'day' đã có trong danh sách, tiếp tục xử lý
            i = new_all_days_data['day'].index(day)
            daily_periodic_df = new_all_days_data['daily_periodic_df'][i]
            for col in daily_periodic_columns:
                daily_periodic_df[col].append(imputed_entry[col])
        
        new_daily_periodic_dfs = []
        for daily_periodic_data in new_all_days_data['daily_periodic_df']:
            daily_periodic_df = pd.DataFrame(daily_periodic_data)
            daily_periodic_df.set_index('period', inplace=True)
            new_daily_periodic_dfs.append(daily_periodic_df)

        new_all_days_data['daily_periodic_df'] = new_daily_periodic_dfs
        imputed_all_days_df = pd.DataFrame(new_all_days_data)
        imputed_all_days_df.set_index('day', inplace=True)
        return imputed_all_days_df

    def process(self):
        df = self._preprocess_periodically_aggregated_values()
        df = self._impute_periodically_aggregated_values(df)
        return df
