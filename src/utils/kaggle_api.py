from kaggle.api.kaggle_api_extended import KaggleApi # type: ignore
import traceback

_kaggle_api = None
def get_kaggle_api() -> KaggleApi:
    global _kaggle_api
    if _kaggle_api is None:
        _kaggle_api = KaggleApi()
        try:
            _kaggle_api.authenticate()
        except:
            traceback.print_exc()
            print(f"Failed to authenticate with Kaggle API. Please check your Kaggle API key.")
            print("See how to get it at <https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md#api-credentials>")
    return _kaggle_api
