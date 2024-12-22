from typing import override

import pandas as pd
from .BasePreprocessor import BasePreprocessor

class BasicTabularFeatureEngineeringPreprocessor(BasePreprocessor):
    """
    Before this: TabularDataLoader
    """

    @override
    def do_load_parameters(self, parameters: dict[str, str]) -> None:
        pass

    def process(self, dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
        df = dfs[0]
        season_cols = [col for col in df.columns if 'Season' in col]
        df = df.drop(season_cols, axis=1) 
        df['BMI_Age'] = df['Physical-BMI'] * df['Basic_Demos-Age']
        df['Internet_Hours_Age'] = df['PreInt_EduHx-computerinternet_hoursday'] * df['Basic_Demos-Age']
        df['BMI_Internet_Hours'] = df['Physical-BMI'] * df['PreInt_EduHx-computerinternet_hoursday']
        df['BFP_BMI'] = df['BIA-BIA_Fat'] / df['BIA-BIA_BMI']
        df['FFMI_BFP'] = df['BIA-BIA_FFMI'] / df['BIA-BIA_Fat']
        df['FMI_BFP'] = df['BIA-BIA_FMI'] / df['BIA-BIA_Fat']
        df['LST_TBW'] = df['BIA-BIA_LST'] / df['BIA-BIA_TBW']
        df['BFP_BMR'] = df['BIA-BIA_Fat'] * df['BIA-BIA_BMR']
        df['BFP_DEE'] = df['BIA-BIA_Fat'] * df['BIA-BIA_DEE']
        df['BMR_Weight'] = df['BIA-BIA_BMR'] / df['Physical-Weight']
        df['DEE_Weight'] = df['BIA-BIA_DEE'] / df['Physical-Weight']
        df['SMM_Height'] = df['BIA-BIA_SMM'] / df['Physical-Height']
        df['Muscle_to_Fat'] = df['BIA-BIA_SMM'] / df['BIA-BIA_FMI']
        df['Hydration_Status'] = df['BIA-BIA_TBW'] / df['Physical-Weight']
        df['ICW_TBW'] = df['BIA-BIA_ICW'] / df['BIA-BIA_TBW']
        df['BMI_PHR'] = df['Physical-BMI'] * df['Physical-HeartRate']
        return [df]
