from typing import Any, cast
import os
import json
from datetime import datetime
import shutil

from src.env import DATA_DIR
from ..DataSource import DataSource
from .DataPullHistoryEntry import DataPullHistoryEntry
from .DataPullHistory import DataPullHistory

class DataPuller:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.data_path_prefix = os.path.join(DATA_DIR, "pulled_data")
        self.data_pull_history_file_path = os.path.join(DATA_DIR, "data_pull_history.json")
        self._load_history()

    def get_data_dir(self, history: DataPullHistoryEntry) -> str:
        return os.path.join(self.data_path_prefix, str(history.to_json_serializable()["when"]), history.type, history.name)
    
    def pull_data(self, source: DataSource) -> DataPullHistoryEntry:
        entry = self.find_latest_pull(source)
        if entry is not None:
            print(f"Data for {source.type}/{source.name} was already pulled at {entry.when}")
            return entry
        return self._pull_data(source)
    
    def force_repull_data(self, source: DataSource) -> DataPullHistoryEntry:
        self.delete_data(source)
        return self._pull_data(source)

    def get_history(self) -> list[DataPullHistoryEntry]:
        return [h for h in self._history]
    
    def find_latest_pull(self, source: DataSource) -> DataPullHistoryEntry | None:
        for entry in reversed(self._history):
            if entry.type == source.type and entry.name == source.name:
                return entry
        return None
    
    def require_data(self, source: DataSource) -> DataPullHistoryEntry:
        entry = self.find_latest_pull(source)
        if entry is None:
            return self._pull_data(source)
        return entry
    
    def delete_data(self, source: DataSource):
        entry = self.find_latest_pull(source)
        if entry is None:
            return
        data_location = os.path.join(self.data_path_prefix, str(entry.to_json_serializable()["when"]), source.type, source.name)
        if os.path.exists(data_location):
            if os.path.isdir(data_location):
                shutil.rmtree(data_location)
            elif os.path.isfile(data_location):
                os.remove(data_location)
            else:
                raise RuntimeError(f"Unexpected file type at {data_location} - neither a file nor a directory?")
        else:
            print(f"Data location {data_location} does not exist")
        self._remove_history_entry(entry)

    def _load_history(self):
        if not os.path.exists(self.data_pull_history_file_path):
            history = []
        else:
            with open(self.data_pull_history_file_path, "r") as f:
                history = json.load(f)
        
        if not isinstance(history, list):
            raise ValueError("Expected a list of history entries")

        self._history = DataPullHistory.from_json_serializable(cast(list[Any], history))
        self._history.sort(key=lambda entry: entry.when)
    
    def __save_history(self):
        with open(self.data_pull_history_file_path, "w") as f:
            json.dump([{"type": entry.type, "name": entry.name, "when": str(int(entry.when.timestamp()))} for entry in self._history], f)
    
    def _add_history_entry(self, entry: DataPullHistoryEntry) -> DataPullHistoryEntry:
        self._history.append(entry)
        self.__save_history()
        return entry
    
    def _remove_history_entry(self, entry: DataPullHistoryEntry) -> DataPullHistoryEntry:
        self._history.remove(entry)
        self.__save_history()
        return entry

    def _pull_data(self, source: DataSource) -> DataPullHistoryEntry:
        # if source not in DataSource.list():
        #     raise ValueError(f"Data source {source.type}/{source.name} does not exist")
        entry = DataPullHistoryEntry(source.type, source.name, datetime.now())
        data_dir = os.path.join(self.data_path_prefix, str(entry.to_json_serializable()["when"]), source.type, source.name)
        source.do_get_data(data_dir)
        self._add_history_entry(entry)
        return entry
