class BaseDataSource:
    def __init__(self, type: str, name: str):
        self.type = type
        self.name = name
    
    def do_get_data(self, data_dir: str) -> None:
        """
        A subclass must override this method to get data from a source and put it in the directory ``data_dir``.
        """
        raise NotImplementedError
