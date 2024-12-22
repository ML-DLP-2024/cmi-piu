from .BaseTrainer import BaseTrainer
from .Trainer1 import Trainer1

from typing import Any, override
import pandas as pd

class Trainer(BaseTrainer):
    CATALOG: dict[str,
        type[Trainer1]
    ] = {
        "trainer1": Trainer1,
    }

    def __init__(self, name: str):
        if name in self.CATALOG:
            self.trainer = self.CATALOG[name]()
        else:
            raise ValueError(f"No trainer named {name} - available options are {self.CATALOG.keys()}")
        super().__init__(name)
    
    @override
    def do_train(self, train: pd.DataFrame, test: pd.DataFrame, model_class: Any) -> pd.DataFrame:
        return self.trainer.do_train(train, test, model_class)

