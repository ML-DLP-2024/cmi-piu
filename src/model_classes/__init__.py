from typing import Any
from .BaseModelClass import BaseModelClass
from .Ensemble1 import Ensemble1
from .Stacking1 import Stacking1
from .Ensemble2 import Ensemble2
from .Stacking2 import Stacking2
from .RandomEnsemble import RandomEnsemble

class ModelClass(BaseModelClass):
    CATALOG: dict[str, 
        type[Ensemble1] | type[Stacking1] | type[Ensemble2] | type[Stacking2] | type[RandomEnsemble]
    ] = {
        "ensemble1": Ensemble1,
        "stacking1": Stacking1,
        "ensemble2": Ensemble2,
        "stacking2": Stacking2,
        "random_ensemble": RandomEnsemble,
    }

    def __init__(self, name: str, parameters: dict[str, str]):
        if name in self.CATALOG:
            self.model = self.CATALOG[name](parameters)
        else:
            raise ValueError(f"No model named {name} - available options are {self.CATALOG.keys()}")
        super().__init__(name, parameters)
    
    def do_create(self, parameters: dict[str, str]) -> tuple[Any, dict[str, str]]:
        return self.model.do_create(parameters)
