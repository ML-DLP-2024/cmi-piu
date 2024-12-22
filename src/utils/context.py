from typing import Any

class DataProcessingContext(dict[str, Any]):
    _context = None

    @classmethod
    def get_instance(cls) -> 'DataProcessingContext':
        cls._context = cls._context or DataProcessingContext()
        return cls._context
