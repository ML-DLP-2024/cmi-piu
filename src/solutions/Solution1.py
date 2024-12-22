from ..data.sources import DataSource
from ..data.preprocessors import Preprocessor
from ..data.pipelines import BasePipeline, Pipeline, PipelinePreviousStage, PipelinesRunner
from ..model_classes import ModelClass
from ..trainers import Trainer
from .BaseSolution import BaseSolution
import pandas as pd

def namspace(prefix: str, parameters: dict[str, str]) -> dict[str, str]:
    return {f"{prefix}_{k}": v for k, v in parameters.items()}

def denamespace(prefix: str, parameters: dict[str, str]) -> dict[str, str]:
    return {k[len(prefix)+1:]: v for k, v in parameters.items() if k.startswith(prefix)}

class Solution1(BaseSolution):
    def __init__(self) -> None:
        super().__init__('solution1')
    
    def do_run(self, data_source: DataSource, parameters: dict[str, str]) -> tuple[pd.DataFrame, dict[str, str]]:
        pipelines : list[BasePipeline] = [
            Pipeline("0", PipelinePreviousStage('loader', ['describing_timeseries']), []),

            Pipeline("0.1", PipelinePreviousStage('loader', ['tabular']), []),

            Pipeline("1", PipelinePreviousStage('pipelines', ['0']), [
                Preprocessor("autoencoder", {
                    "encoding_dim": "50",
                    "epochs": "50",
                    "batch_size": "32",
                }),
            ]),

            Pipeline("2", PipelinePreviousStage('pipelines', ['0.1']), [
                Preprocessor("basic_feature_engineering", {}),
            ]),

            Pipeline("3", PipelinePreviousStage('loader', ['alfe']), []),

            Pipeline("4", PipelinePreviousStage('pipelines', ['1', '2', '3']), [
                Preprocessor("merge", {}),
            ]),

            Pipeline("5", PipelinePreviousStage('pipelines', ['4']), [
                Preprocessor("feature_selection", {}),
            ]),

            Pipeline("6", PipelinePreviousStage('pipelines', ['5']), [
                Preprocessor("inf_to_nan", {}),
            ]),

            Pipeline("7", PipelinePreviousStage('pipelines', ['6']), [
                Preprocessor("drop_na", {}),
            ]),

            Pipeline("4.2", PipelinePreviousStage('pipelines', ['0', '0.1']), [
                Preprocessor("merge", {}),
                Preprocessor("feature_selection_2", {}),
            ]),

            Pipeline("4_test", PipelinePreviousStage('pipelines', ['1', '2', '3']), [
                Preprocessor("union_merge", {}),
                Preprocessor("feature_selection", {}),
            ]),

            Pipeline("4.2_test", PipelinePreviousStage('pipelines', ['0', '0.1']), [
                Preprocessor("union_merge", {}),
                Preprocessor("feature_selection_2", {}),
            ]),
        ]

        train_preprocessed = PipelinesRunner(pipelines, data_source).run('train')
        test_preprocessed = PipelinesRunner(pipelines, data_source).run('test')

        ensemble1, params_ensemble1 = ModelClass('ensemble1', denamespace('ensemble1_', parameters)).create()
        stacking1, params_stacking1 = ModelClass('stacking1', denamespace('stacking1_', parameters)).create()
        ensemble2, params_ensemble2 = ModelClass('ensemble2', denamespace('ensemble2_', parameters)).create()
        stacking2, params_stacking2 = ModelClass('stacking2', denamespace('stacking2_', parameters)).create()
        random_ensemble, params_random_ensemble = ModelClass('random_ensemble', denamespace('random_ensemble_', parameters)).create()

        # output_pipeline_names = [
        #     '7' # for Ensemble1 and Stacking1 (training)
        #     '4.2' # for Ensemble2, Stacking2, RandomEnsemble (training)
        #     '4.2_test' # for Ensemble2, Stacking2, RandomEnsemble (prediction)
        #     '4_test' # for Ensemble1 and Stacking1 (prediction)
        # ]

        trainer1 = Trainer('trainer1')
        print("Training Ensemble1")
        sub1 = trainer1.train(train_preprocessed['7'][0], test_preprocessed['4_test'][0], ensemble1)
        print("Training Stacking1")
        sub2 = trainer1.train(train_preprocessed['7'][0], test_preprocessed['4_test'][0], stacking1)
        
        print("Training Ensemble2")
        sub3 = trainer1.train(train_preprocessed['4.2'][0], test_preprocessed['4.2_test'][0], ensemble2)
        print("Training Stacking2")
        sub4 = trainer1.train(train_preprocessed['4.2'][0], test_preprocessed['4.2_test'][0], stacking2)
        print("Training RandomEnsemble")
        sub5 = trainer1.train(train_preprocessed['4.2'][0], test_preprocessed['4.2_test'][0], random_ensemble)

        print("Major Voting...")

        sub1 = sub1.sort_values(by='id').reset_index(drop=True) # type: ignore
        sub2 = sub2.sort_values(by='id').reset_index(drop=True) # type: ignore
        sub3 = sub3.sort_values(by='id').reset_index(drop=True) # type: ignore
        sub4 = sub4.sort_values(by='id').reset_index(drop=True) # type: ignore
        sub5 = sub5.sort_values(by='id').reset_index(drop=True) # type: ignore
        combined = pd.DataFrame({
            'id': sub1['id'],
            'sii_1': sub1['sii'],
            'sii_2': sub2['sii'],
            'sii_3': sub3['sii'],
            'sii_4': sub4['sii'],
            'sii_5': sub5['sii']
        })

        def majority_vote(row): # type: ignore
            return row.mode()[0] # type: ignore

        combined['final_sii'] = combined[['sii_1', 'sii_2', 'sii_3', 'sii_4', 'sii_5']].apply(majority_vote, axis=1) # type: ignore

        final_submission = combined[['id', 'final_sii']].rename(columns={'final_sii': 'sii'})

        new_config = namspace('ensemble1', params_ensemble1)
        new_config.update(namspace('stacking1', params_stacking1))
        new_config.update(namspace('ensemble2', params_ensemble2))
        new_config.update(namspace('stacking2', params_stacking2))
        new_config.update(namspace('random_ensemble', params_random_ensemble))

        return final_submission, new_config
