from typing import Any
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

class SkillfulImputer:
    """Uses an sklearn imputer. Takes into account the categorical columns. Leaves NaN columns intact.
    (NaN columns are columns that contain only NaN values.)
    """

    def __init__(self, imputer: Any):
        """
        imputer   - Must be an instance of an sklearn imputer, e.g.
                    KNNImputer(n_neighbors=2)
        """
        self.imputer = imputer

    def fit_transform(self, df: pd.DataFrame, categorical_columns: list[str], **kwargs: dict[str, Any]) -> pd.DataFrame:
        df = df.copy()
        categorical_columns = list(categorical_columns)

        # A. Identify NaN columns
        nan_columns = df.columns[
            df.isna().all() # type: ignore
        ].to_list()
        
        # B. Convert categorical to numerical (temporarily)
        label_encoders: dict[str, LabelEncoder] = {}
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = df[col].astype(str) # convert to string to handle NaN
            df[col] = le.fit_transform(
                df[col] # type: ignore
            )
            label_encoders[col] = le # Save the encoder to decode later!
        # C. Impute!
        imputer = self.imputer
        imputed_data = imputer.fit_transform(df)
        imputed_df = pd.DataFrame(
            imputed_data,
            columns=df.columns,
            **kwargs # type: ignore
        )
        # D. Convert the categorical columns back
        for col in categorical_columns:
            imputed_df[col] = imputed_df[
                col # type: ignore
            ].round(0).astype(int)
            imputed_df[col] = label_encoders[
                col
            ].inverse_transform( # type: ignore
                imputed_df[col] # type: ignore
            )
        # E. Restore the NaN values
        for col in nan_columns:
            imputed_df[col] = np.nan
        return imputed_df
