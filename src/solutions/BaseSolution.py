import pandas as pd
from ..data.sources import DataSource
import pandas as pd
import json

class BaseSolution:
    def __init__(self, name: str):
        self.name = name
    
    def do_run(self, data_source: DataSource, parameters: dict[str, str]) -> tuple[pd.DataFrame, dict[str, str]]:
        """Returns the prediction as well as new hyper-parameters"""
        raise NotImplementedError()
    
    def run(self, data_source: DataSource) -> pd.DataFrame:
        import os
        if not os.path.exists('config.json'):
            with open('config.json', 'w') as f:
                json.dump({}, f)
        with open('config.json', 'r') as f:
            config = json.load(f)
        try:
            pred, config = self.do_run(data_source, config)
            return pred
        finally:
            with open('config.json', 'w') as f:
                json.dump(config, f)
