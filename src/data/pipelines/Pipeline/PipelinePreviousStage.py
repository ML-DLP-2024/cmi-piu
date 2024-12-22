from typing import Literal, TypedDict, Any

class PipelinePreviousStageTypedDict(TypedDict):
    type: Literal['pipelines', 'loader']
    names: list[str]

class PipelinePreviousStage:
    def __init__(self, type: Literal['pipelines', 'loader'], names: list[str]) -> None:
        if type == 'pipelines':
            if len(names) == 0:
                raise ValueError("PipelinePreviousStage with type 'pipelines' must have at least one pipeline name")
            self.names = [str(n) for n in names]
        elif type == 'loader':
            if len(names) != 1:
                raise ValueError("PipelinePreviousStage with type 'loader' must have exactly one loader name")
            self.names = [str(names[0])]
        else:
            raise ValueError(f"Unknown pipeline previous stage type: {type}")
        self.type: Literal['pipelines', 'loader'] = type
    
    def to_json_serializable(self) -> PipelinePreviousStageTypedDict:
        return {
            'type': self.type,
            'names': self.names
        }
    
    @classmethod
    def from_json_serializable(cls, obj: Any) -> 'PipelinePreviousStage':
        if not isinstance(obj, dict):
            raise ValueError("PipelinePreviousStage.from_json_serializable expects a dictionary")
        if 'type' not in obj or 'names' not in obj:
            raise ValueError("PipelinePreviousStage.from_json_serializable expects a dictionary with 'type' and 'names' keys")
        if not isinstance(obj['type'], str):
            raise ValueError("PipelinePreviousStage.from_json_serializable expects 'type' to be a string")
        if not isinstance(obj['names'], list):
            raise ValueError("PipelinePreviousStage.from_json_serializable expects 'names' to be a list")
        if obj['type'] != 'pipelines' and obj['type'] != 'loader':
            raise ValueError(f"Unknown pipeline previous stage type: {obj['type']}")
        names: Any = obj['names']
        type: Literal['pipelines', 'loader'] = obj['type']
        return cls(type, [str(n) for n in names])
