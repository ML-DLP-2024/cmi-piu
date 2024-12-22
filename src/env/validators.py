import os

class EnvKeyInvalidOrMissing(Exception):
    def __init__(self, key: str, expecting: str|None) -> None:
        expecting = f" Expecting {expecting}" if expecting else ""
        super().__init__(f"Environment key '{key}' is invalid or missing.{expecting}")

def read_string_env(key: str, expecting: str = "a text string") -> str:
    value = os.getenv(key)
    if not value:
        raise EnvKeyInvalidOrMissing(key, expecting)
    return value

def read_int_env(key: str, expecting: str = "an integer") -> int:
    value = read_string_env(key, expecting)
    try:
        return int(value)
    except ValueError:
        raise EnvKeyInvalidOrMissing(key, expecting)
