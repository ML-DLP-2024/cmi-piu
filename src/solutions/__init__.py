from pandas.core.api import DataFrame as DataFrame
from src.data.sources.DataSource import DataSource
from .BaseSolution import BaseSolution
from .Solution1 import Solution1

class Solution(BaseSolution):
    CATALOG: dict[str, type[Solution1]] = {
        'solution1': Solution1,
    }

    @classmethod
    def list(cls):
        return cls.CATALOG.keys()
    
    def __init__(self, name: str):
        if name in self.CATALOG:
            self.solution = self.CATALOG[name]()
        else:
            raise ValueError(f"No solution named {name} - available options are {self.CATALOG.keys()}")
        super().__init__(name)
    
    def do_run(self, data_source: DataSource, parameters: dict[str, str]) -> tuple[DataFrame, dict[str, str]]:
        return self.solution.do_run(data_source, parameters)
