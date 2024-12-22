from typing import TypedDict
import pandas as pd
from .PipelinePreviousStage import PipelinePreviousStage, PipelinePreviousStageTypedDict
from ...preprocessors.Preprocessor import Preprocessor
from ...preprocessors.Preprocessor import PreprocessorTypedDict

class BasePipelineTypedDict(TypedDict):
    name: str
    prev: PipelinePreviousStageTypedDict
    preprocessors: list[PreprocessorTypedDict]

class BasePipeline:
    def __init__(self, name: str, prev: PipelinePreviousStage, preprocessors: list[Preprocessor]):
        self.name = name
        self.prev = prev
        self.pps: list[Preprocessor] = list(preprocessors)
    
    def run_preprocessors(self, dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
        for pp in self.pps:
            dfs = pp.process(dfs)
        return dfs
    
    def to_json_serializable(self) -> BasePipelineTypedDict:
        return {
            'name': self.name,
            'prev': self.prev.to_json_serializable(),
            'preprocessors': [pp.to_json_serializable() for pp in self.pps]
        }
    
    @classmethod
    def from_json_serializable(cls, data: BasePipelineTypedDict) -> 'BasePipeline':
        if not isinstance(data, dict): # type: ignore
            raise ValueError(f"Expected a dictionary, got {type(data)}")

        try:
            name = data['name']
            prev = data['prev']
            preprocessors = data['preprocessors']
        except KeyError as e:
            raise ValueError(f"Missing key {e}")

        return cls(
            str(name),
            PipelinePreviousStage.from_json_serializable(prev),
            [Preprocessor.from_json_serializable(pp) for pp in preprocessors]
        )
