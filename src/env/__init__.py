from dotenv import load_dotenv
load_dotenv()

from .validators import read_string_env

DATA_DIR = read_string_env("DATA_DIR")
