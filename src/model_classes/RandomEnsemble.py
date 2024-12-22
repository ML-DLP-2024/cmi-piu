from typing import Any
from .BaseModelClass import BaseModelClass

from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor # type: ignore
from catboost import CatBoostRegressor # type: ignore
from sklearn.pipeline import Pipeline

class RandomEnsemble(BaseModelClass):
    def __init__(self, parameters: dict[str, str]):
        super().__init__("random_ensemble", parameters)
    def do_create(self, parameters: dict[str, str]) -> tuple[Any, dict[str, str]]:
        SEED = 42

        imputer = SimpleImputer(strategy='median')

        ensemble = VotingRegressor(estimators=[
            ('lgb',    Pipeline(steps=[('imputer', imputer), ('regressor', LGBMRegressor(random_state=SEED))])),
            ('xgb',    Pipeline(steps=[('imputer', imputer), ('regressor', XGBRegressor(random_state=SEED))])),
            ('cat',    Pipeline(steps=[('imputer', imputer), ('regressor', CatBoostRegressor(random_state=SEED, silent=True))])),
            ('rf',     Pipeline(steps=[('imputer', imputer), ('regressor', RandomForestRegressor(random_state=SEED))])),
            ('gb',     Pipeline(steps=[('imputer', imputer), ('regressor', GradientBoostingRegressor(random_state=SEED))])),
            #('tabnet', Pipeline(steps=[('imputer', imputer), ('regressor', TabNetWrapper(**TabNet_Params))])),
            #('odt',    Pipeline(steps=[('imputer', imputer), ('regressor', ObliqueDecisionTreeRegressor(**ODT_Params))])),
        ])

        return ensemble, {}
