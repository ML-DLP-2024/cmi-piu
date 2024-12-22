from typing import Any

class BaseModelClass:
    def do_create(self, parameters: dict[str, str]) -> tuple[Any, dict[str, str]]:
        """Creates the model class and returns it along with the new (hyper) parameters."""
        raise NotImplementedError
    
    def __init__(self, name: str, parameters: dict[str, str]):
        self.name = name
        self.parameters = dict(parameters)
    
    def create(self) -> Any:
        """Creates the model class and returns it along with the new (hyper) parameters."""
        return self.do_create(self.parameters)
