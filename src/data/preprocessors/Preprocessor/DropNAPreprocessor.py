from .BasePreprocessor import BasePreprocessor
import pandas as pd

class DropNAPreprocessor(BasePreprocessor):
    """
    Before this: any dfs!
    """

    def do_load_parameters(self, parameters: dict[str, str]) -> None:
        pass

    def process(self, dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
        df = dfs[0]
        df = df.dropna(thresh=10, axis=0) # type: ignore
        return [df]