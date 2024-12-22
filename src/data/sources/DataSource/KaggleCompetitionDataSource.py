from .BaseDataSource import BaseDataSource
from src.utils.kaggle_api import get_kaggle_api
import os
import zipfile

class KaggleCompetitionDataSource(BaseDataSource):
    def __init__(self, name: str):
        super().__init__('kaggle', name)
    
    def do_get_data(self, data_dir: str) -> None:
        api = get_kaggle_api()
        data_dir = os.path.normpath(os.path.join(data_dir, self.name, "../"))
        print(data_dir)
        api.competition_download_cli( # type: ignore
            competition=self.name, path=data_dir
        )
        file_path = os.path.join(data_dir, self.name + ".zip")

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        os.remove(file_path)
