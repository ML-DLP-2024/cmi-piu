from typing import Any, TypedDict
from .BaseDataSource import BaseDataSource
from .KaggleCompetitionDataSource import KaggleCompetitionDataSource

class DataSourceDict(TypedDict):
    type: str
    name: str

class DataSource(BaseDataSource):
    CATALOG = {
        "kaggle": KaggleCompetitionDataSource
    }

    def __init__(self, source_type: str, source_name: str):
        if source_type not in self.CATALOG:
            raise ValueError(f"Unknown source type: {source_type}")
        self.type = source_type
        self.name = source_name
        self.obj = self.CATALOG[source_type](source_name)
    
    def do_get_data(self, data_dir: str) -> None:
        return self.obj.do_get_data(data_dir)
    
    def __eq__(self, value: object) -> bool:
        if isinstance(value, DataSource):
            return self.type == value.type and self.name == value.name
        return super().__eq__(value)
    
    def to_json_serializable(self) -> DataSourceDict:
        return {
            "type": self.type,
            "name": self.name,
        }
    
    @classmethod
    def from_json_serializable(cls, d: Any) -> "DataSource":
        if not isinstance(d, dict):
            raise ValueError(f"Expected dict, got {type(d)}")
        source_type: Any = d["type"]
        source_name: Any = d["name"]
        if not isinstance(source_type, str) or not isinstance(source_name, str):
            raise ValueError(f"Expected 'type' and 'name' to be strings")
        return cls(source_type, source_name)
    
    @staticmethod
    def list() -> list[BaseDataSource]:
        return [
            KaggleCompetitionDataSource("child-mind-institute-problematic-internet-use"),
        ]
