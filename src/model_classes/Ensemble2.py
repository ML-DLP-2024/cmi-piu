from typing import Any
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor # type: ignore
from catboost import CatBoostRegressor # type: ignore
from sklearn.ensemble import VotingRegressor
from .BaseModelClass import BaseModelClass

DEFAULT_SEED = 42

DEFAULT_CAT_FEATURES = ['Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 
          'Fitness_Endurance-Season', 'FGC-Season', 'BIA-Season', 
          'PAQ_A-Season', 'PAQ_C-Season', 'SDS-Season', 'PreInt_EduHx-Season']

class Ensemble2(BaseModelClass):
    def __init__(self, parameters: dict[str, str]):
        super().__init__("ensemble2", parameters)
    
    def do_create(self, parameters: dict[str, str]) -> tuple[Any, dict[str, str]]:
        LGBM_Params: dict[str, Any] = {
            'learning_rate': float(parameters.get('lgbm_learning_rate', 0.046)),
            'max_depth': int(parameters.get('lgbm_max_depth', 12)),
            'num_leaves': int(parameters.get('lgbm_num_leaves', 478)),
            'min_data_in_leaf': int(parameters.get('lgbm_min_data_in_leaf', 13)),
            'feature_fraction': float(parameters.get('lgbm_feature_fraction', 0.893)),
            'bagging_fraction': float(parameters.get('lgbm_bagging_fraction', 0.784)),
            'bagging_freq': int(parameters.get('lgbm_bagging_freq', 4)),
            'lambda_l1': float(parameters.get('lgbm_lambda_l1', 10)),  # Increased from 6.59
            'lambda_l2': float(parameters.get('lgbm_lambda_l2', 0.01)),  # Increased from 2.68e-06
            'random_state': int(parameters.get('lgbm_random_state', DEFAULT_SEED)),
            'n_estimators': int(parameters.get('lgbm_n_estimators', 300)),
            'verbose': -1,
            'device': 'cpu',
        }

        XGB_Params: dict[str, Any] = {
            'learning_rate': float(parameters.get('xgb_learning_rate', 0.05)),
            'max_depth': int(parameters.get('xgb_max_depth', 6)),
            'n_estimators': int(parameters.get('xgb_n_estimators', 200)),
            'subsample': float(parameters.get('xgb_subsample', 0.8)),
            'colsample_bytree': float(parameters.get('xgb_colsample_bytree', 0.8)),
            'reg_alpha': float(parameters.get('xgb_reg_alpha', 1)),  # Increased from 0.1
            'reg_lambda': float(parameters.get('xgb_reg_lambda', 5)),  # Increased from 1
            'random_state': int(parameters.get('xgb_random_state', DEFAULT_SEED)),
            'tree_method': 'gpu_hist'
        }

        CatBoost_Params: dict[str, Any] = {
            'learning_rate': float(parameters.get('catboost_learning_rate', 0.05)),
            'depth': int(parameters.get('catboost_depth', 6)),
            'iterations': int(parameters.get('catboost_iterations', 200)),
            'random_seed': int(parameters.get('catboost_random_seed', DEFAULT_SEED)),
            'verbose': 0,
            'cat_features': parameters.get('cat_features', DEFAULT_CAT_FEATURES),
            'l2_leaf_reg': float(parameters.get('catboost_l2_leaf_reg', 10)),  # Increase this value
            'task_type': 'GPU',
        }

        # Create model instances
        Light = LGBMRegressor(**LGBM_Params)
        XGB_Model = XGBRegressor(**XGB_Params)
        CatBoost_Model = CatBoostRegressor(**CatBoost_Params)

        # Combine models using Voting Regressor
        voting_model = VotingRegressor(estimators=[ # type: ignore
            ('lightgbm', Light),
            ('xgboost', XGB_Model),
            ('catboost', CatBoost_Model),
        ]) # , weights=[4.0,4.0,5.0]

        new_parameters = {
            'lgbm_learning_rate': str(LGBM_Params['learning_rate']),
            'lgbm_max_depth': str(LGBM_Params['max_depth']),
            'lgbm_num_leaves': str(LGBM_Params['num_leaves']),
            'lgbm_min_data_in_leaf': str(LGBM_Params['min_data_in_leaf']),
            'lgbm_feature_fraction': str(LGBM_Params['feature_fraction']),
            'lgbm_bagging_fraction': str(LGBM_Params['bagging_fraction']),
            'lgbm_bagging_freq': str(LGBM_Params['bagging_freq']),
            'lgbm_lambda_l1': str(LGBM_Params['lambda_l1']),
            'lgbm_lambda_l2': str(LGBM_Params['lambda_l2']),
            'lgbm_random_state': str(LGBM_Params['random_state']),
            'lgbm_n_estimators': str(LGBM_Params['n_estimators']),
            'xgb_learning_rate': str(XGB_Params['learning_rate']),
            'xgb_max_depth': str(XGB_Params['max_depth']),
            'xgb_n_estimators': str(XGB_Params['n_estimators']),
            'xgb_subsample': str(XGB_Params['subsample']),
            'xgb_colsample_bytree': str(XGB_Params['colsample_bytree']),
            'xgb_reg_alpha': str(XGB_Params['reg_alpha']),
            'xgb_reg_lambda': str(XGB_Params['reg_lambda']),
            'xgb_random_state': str(XGB_Params['random_state']),
            'catboost_learning_rate': str(CatBoost_Params['learning_rate']),
            'catboost_depth': str(CatBoost_Params['depth']),
            'catboost_iterations': str(CatBoost_Params['iterations']),
            'catboost_random_seed': str(CatBoost_Params['random_seed']),
            'catboost_l2_leaf_reg': str(CatBoost_Params['l2_leaf_reg']),
        }

        return voting_model, new_parameters
