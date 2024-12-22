from typing import Any
from .DataPullHistoryEntry import DataPullHistoryEntry, DataPullHistoryEntryDict

class DataPullHistory(list[DataPullHistoryEntry]):
    def append(self, entry: DataPullHistoryEntry):
        super().append(entry)
    
    def to_json_serializable(self) -> list[DataPullHistoryEntryDict]:
        return [entry.to_json_serializable() for entry in self]
    
    @classmethod
    def from_json_serializable(cls, history: list[Any]):
        return cls([DataPullHistoryEntry.from_json_serializable(entry) for entry in history])
