from typing import Literal
from .Pipeline import Pipeline
import pandas as pd
from src.data.loaders.DataLoader import DataLoader
from src.data.sources.DataPuller import DataPuller
from src.data.sources.DataSource import DataSource

class PipelinesRunner:
    def __init__(self, pipelines: list[Pipeline], source: DataSource, output_pipeline_name: str) -> None:
        self.pipelines = pipelines
        self.source = source
        self.output_pipeline_name = output_pipeline_name

    def run(self, dataset: Literal['train', 'test']):
        puller = DataPuller()
        data_dir = puller.get_data_dir(puller.require_data(self.source))

        def order_of_execution(p: Pipeline) -> int:
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

                    print(f"Running pipeline {pipeline.name}")
                    for pp in pipeline.pps:
                        dfs = pp.process(dfs)
                pipeline_execution_result_by_name[pipeline.name] = pipeline.run_preprocessors(dfs)
            i += 1
            if i > limit:
                break
        
        if not done:
            raise RuntimeError("Pipeline execution did not converge - there is a loop.")
        
        return pipeline_execution_result_by_name[self.output_pipeline_name]
