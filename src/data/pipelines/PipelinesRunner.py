from typing import Literal
from .Pipeline import BasePipeline
import pandas as pd
from src.data.loaders.DataLoader import DataLoader
from src.data.sources.DataPuller import DataPuller
from src.data.sources.DataSource import DataSource

class PipelinesRunner:
    def __init__(self, pipelines: list[BasePipeline], source: DataSource) -> None:
        """Remember: by default, the last pipeline is the output pipeline"""
        self.pipelines = pipelines
        self.source = source

    def run(self, dataset: Literal['train', 'test']) -> dict[str, list[pd.DataFrame]]:
        puller = DataPuller()
        data_dir = puller.get_data_dir(puller.require_data(self.source))

        def order_of_execution(p: BasePipeline) -> int:
            return 0 if p.prev.type == 'loader' else 1
        
        self.pipelines.sort(key=order_of_execution)

        pipeline_execution_result_by_name: dict[str, list[pd.DataFrame]] = {}
        loader_execution_result_by_name: dict[str, pd.DataFrame] = {}

        limit = len(self.pipelines)
        done = False
        i = 0
        while not done:
            done = True
            for pipeline in self.pipelines:
                if pipeline_execution_result_by_name.get(pipeline.name, None) is not None:
                    # Pipeline already executed
                    continue
                if pipeline.prev.type == 'loader':
                    df = loader_execution_result_by_name.get(pipeline.name, None)
                    if df is None:
                        done = False
                        loader_name = pipeline.prev.names[0]
                        loader = DataLoader(loader_name, data_dir)
                        print(f"Running loader {loader_name} for dataset {dataset}")
                        df = loader_execution_result_by_name[pipeline.name] = loader.load(dataset)
                    dfs = [df]
                else:
                    dfs: list[pd.DataFrame] = []
                    run_later = False
                    for prev_pipeline_name in pipeline.prev.names:
                        more_dfs = pipeline_execution_result_by_name.get(prev_pipeline_name, None)
                        if more_dfs is None:
                            # A previous pipeline is not ready.
                            done = False
                            run_later = True
                            break
                        dfs.extend(more_dfs)
                    if run_later:
                        continue

                    for pp in pipeline.pps:
                        dfs = pp.process(dfs)
                print(f"Running pipeline {pipeline.name}")
                pipeline_execution_result_by_name[pipeline.name] = pipeline.run_preprocessors(dfs)
            i += 1
            if i > limit:
                break
        
        if not done:
            raise RuntimeError("Pipeline execution did not converge - there is a loop.")
        
        return pipeline_execution_result_by_name
