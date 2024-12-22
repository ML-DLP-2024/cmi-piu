from typing import override, Any, TypedDict
import pandas as pd
from .BasePreprocessor import BasePreprocessor

from .AutoEncoderPreprocessor import AutoEncoderPreprocessor
from .BasicTabularFeatureEngineeringPreprocessor import BasicTabularFeatureEngineeringPreprocessor
from .MergePreprocessor import MergePreprocessor
from .UnionMergePreprocessor import UnionMergePreprocessor
from .FeatureSelectionPreprocessor import FeatureSelectionPreprocessor
from .InfToNaNPreprocessor import InfToNanPreprocessor
from .DropNAPreprocessor import DropNAPreprocessor
from .FeatureSelection2Preprocessor import FeatureSelection2Preprocessor

class PreprocessorTypedDict(TypedDict):
    name: str
    parameters: dict[str, str]

class Preprocessor(BasePreprocessor):
    CATALOG: dict[str,
        type[AutoEncoderPreprocessor]
        | type[BasicTabularFeatureEngineeringPreprocessor]
        | type[MergePreprocessor]
        | type[FeatureSelectionPreprocessor]
        | type[InfToNanPreprocessor]
        | type[DropNAPreprocessor]
        | type[FeatureSelection2Preprocessor]
        | type[UnionMergePreprocessor]
    ] = {
        "autoencoder": AutoEncoderPreprocessor,
        "basic_feature_engineering": BasicTabularFeatureEngineeringPreprocessor,
        "merge": MergePreprocessor,
        "feature_selection": FeatureSelectionPreprocessor,
        "inf_to_nan": InfToNanPreprocessor,
        "drop_na": DropNAPreprocessor,
        "feature_selection_2": FeatureSelection2Preprocessor,
        "union_merge": UnionMergePreprocessor,
    }

    @classmethod
    def get_preprocessor_class(cls, name: str):
        try:
            return cls.CATALOG[name]
        except KeyError:
            raise ValueError(f"No preprocessor named {name} - available options are {cls.CATALOG.keys()}")

    def __init__(self, name: str, parameters: dict[str, str]) -> None:
        if name in self.CATALOG:
            self.preprocessor = self.CATALOG[name](name, parameters)
        else:
            raise ValueError(f"No preprocessor named {name} - available options are {self.CATALOG.keys()}")
        super().__init__(name, parameters)

    @override
    def do_load_parameters(self, parameters: dict[str, str]) -> None:
        return self.preprocessor.do_load_parameters(parameters)

    @override
    def process(self, dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
        return self.preprocessor.process(dfs)

    def to_json_serializable(self) -> PreprocessorTypedDict:
        return {
            'name': self.name,
            'parameters': self.parameters
        }
    
    @classmethod
    def from_json_serializable(cls, d: Any) -> 'Preprocessor':
        name = d['name']
        parameters = d['parameters']
        if name not in cls.CATALOG:
            raise ValueError(f"No preprocessor named {name} - available options are {cls.CATALOG.keys()}")
        return cls(name, parameters)
