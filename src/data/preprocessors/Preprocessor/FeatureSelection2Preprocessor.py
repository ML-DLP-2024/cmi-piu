from typing import override
import pandas as pd
from .BasePreprocessor import BasePreprocessor
from src.utils.context import DataProcessingContext

class FeatureSelection2Preprocessor(BasePreprocessor):
    @override
    def do_load_parameters(self, parameters: dict[str, str]) -> None:
        pass

    def process(self, dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
        featuresCols: list[str] = ['Basic_Demos-Age', 'Basic_Demos-Sex',
            'CGAS-CGAS_Score', 'Physical-BMI',
            'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference',
            'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP',
            'Fitness_Endurance-Max_Stage',
            'Fitness_Endurance-Time_Mins', 'Fitness_Endurance-Time_Sec',
            'FGC-FGC_CU', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND',
            'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD', 'FGC-FGC_GSD_Zone', 'FGC-FGC_PU',
            'FGC-FGC_PU_Zone', 'FGC-FGC_SRL', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR',
            'FGC-FGC_SRR_Zone', 'FGC-FGC_TL', 'FGC-FGC_TL_Zone',
            'BIA-BIA_Activity_Level_num', 'BIA-BIA_BMC', 'BIA-BIA_BMI',
            'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 'BIA-BIA_FFM',
            'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat', 'BIA-BIA_Frame_num',
            'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM',
            'BIA-BIA_TBW', 'PAQ_A-PAQ_A_Total',
            'PAQ_C-PAQ_C_Total', 'SDS-SDS_Total_Raw',
            'SDS-SDS_Total_T',
            'PreInt_EduHx-computerinternet_hoursday', 'sii', 'BMI_Age','Internet_Hours_Age','BMI_Internet_Hours',
            'BFP_BMI', 'FFMI_BFP', 'FMI_BFP', 'LST_TBW', 'BFP_BMR', 'BFP_DEE', 'BMR_Weight', 'DEE_Weight',
            'SMM_Height', 'Muscle_to_Fat', 'Hydration_Status', 'ICW_TBW', 'BMI_PHR'
        ]
        featuresCols.extend( DataProcessingContext.get_instance().get("time_series_cols", []) )

        df = dfs[0]

        return [df[featuresCols]]
