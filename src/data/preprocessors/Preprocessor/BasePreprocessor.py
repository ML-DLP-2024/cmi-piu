import pandas as pd

class BasePreprocessor:
    def __init__(self, name: str, parameters: dict[str, str]) -> None:
        self.name = name
        self.parameters = parameters
        self.do_load_parameters(parameters)

    def do_load_parameters(self, parameters: dict[str, str]) -> None:
        """A subclass can override this method to load parameters from the dictionary.
        In case the preprocessor need no parameters, you do not have to override this method."""
        pass

    def process(self, dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """A subclass must override this method to process the input DataFrame."""
        raise NotImplementedError
