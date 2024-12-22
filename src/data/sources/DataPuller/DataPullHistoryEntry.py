from typing import Any, TypedDict
from datetime import datetime

class DataPullHistoryEntryDict(TypedDict):
    type: str
    name: str
    when: str

class DataPullHistoryEntry:
    def __init__(self, type: str, name: str, when: datetime | int):
        self.type = type
        self.name = name
        if isinstance(when, int):
            self.when = datetime.fromtimestamp(when)
        else:
            self.when = when
    
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, DataPullHistoryEntry):
            return False
        return self.type == value.type and self.name == value.name and self.when == value.when
    
    def to_json_serializable(self) -> DataPullHistoryEntryDict:
        return {"type": self.type, "name": self.name, "when": str(int(self.when.timestamp())) }
    
    @classmethod
    def from_json_serializable(cls, entry: dict[str, Any]):
        if "type" not in entry or "name" not in entry or "when" not in entry:
            raise ValueError("Expected a list of history entries")
        if not isinstance(entry["type"], str) or not isinstance(entry["name"], str) or not isinstance(entry["when"], str):
            raise ValueError("Expected a list of history entries with name, type and datetime (when) fields")
        return cls(entry["type"], entry["name"], int(entry["when"]))
