from typing import override
import pandas as pd
import numpy as np
from .BasePreprocessor import BasePreprocessor

class InfToNanPreprocessor(BasePreprocessor):
    """
    Before this: any dfs!
    """

    @override
    def do_load_parameters(self, parameters: dict[str, str]) -> None:
        pass

    def process(self, dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
        df = dfs[0]
        if np.any(np.isinf(df)):
            df = df.replace([np.inf, -np.inf], np.nan) # type: ignore
        return [df]
