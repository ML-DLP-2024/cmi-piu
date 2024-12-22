from typing import override
import pandas as pd
from .BasePreprocessor import BasePreprocessor

class MergePreprocessor(BasePreprocessor):
    """
    Before this: any dfs!
    """

    @override
    def do_load_parameters(self, parameters: dict[str, str]) -> None:
        pass

    def process(self, dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
        df = dfs.pop(0)
        while len(dfs) > 0:
            df2 = dfs.pop(0)
            df = pd.merge( # type: ignore
                df, df2,
                on='id', how='left'
            )
        return [df]
