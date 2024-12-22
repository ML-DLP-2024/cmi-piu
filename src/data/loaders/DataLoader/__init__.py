from pandas.core.api import DataFrame as DataFrame
from .BaseDataLoader import BaseDataLoader
from .DescribingTimeseriesDataLoader import DescribingTimeseriesDataLoader
from .AlfeIntegratedDataLoader import AlfeIntegratedDataLoader
from .TabularDataLoader import TabularDataLoader
from typing import override

class DataLoader(BaseDataLoader):
    CATALOG: dict[str,
        type[DescribingTimeseriesDataLoader]
        | type[AlfeIntegratedDataLoader]
        | type[TabularDataLoader]
    ] = {
        'describing_timeseries': DescribingTimeseriesDataLoader,
        'alfe': AlfeIntegratedDataLoader,
        'tabular': TabularDataLoader,
    }

    @override
    def __init__(self, name: str, data_dir: str):
        if name in self.CATALOG:
            self.data_loader = self.CATALOG[name](name, data_dir)
        else:
            raise ValueError(f"No data loader named {name} - available options are {self.CATALOG.keys()}")
        super(DataLoader, self).__init__(name, data_dir)
    
    @override
    def do_load(self, dataset: str) -> DataFrame:
        return self.data_loader.do_load(dataset)
