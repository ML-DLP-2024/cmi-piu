import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore FutureWarning from sklearn or pandas or something

import click
from tabulate import tabulate
from typing import Any
import time

@click.group()
def cli():
    pass

@cli.group()
def data():
    pass

@data.command()
def sources():
    from src.data.sources import DataSource
    d: list[dict[str, Any]] = []
    for source in DataSource.list():
        d.append({
            "type": source.type,
            "name": source.name,
        })
    print(tabulate(d, headers="keys", tablefmt="pretty"))

@data.command()
@click.argument("type")
@click.argument("name")
def pull(type: str, name: str):
    from src.data.sources.DataPuller import DataPuller, DataSource
    puller = DataPuller()
    puller.pull_data(DataSource(type, name))

@data.command()
def history():
    from src.data.sources.DataPuller import DataPuller
    puller = DataPuller()
    d: list[dict[str, Any]] = []
    for entry in puller.get_history():
        d.append({
            "when": entry.when.isoformat(),
            "type": entry.type,
            "name": entry.name,
        })
    print(tabulate(d, headers="keys", tablefmt="pretty"))

@data.command
@click.argument("type")
@click.argument("name")
def delete(type: str, name: str):
    from src.data.sources.DataPuller import DataPuller, DataSource
    puller = DataPuller()
    puller.delete_data(DataSource(type, name))

@data.command
def loaders():
    from src.data.loaders.DataLoader import DataLoader
    d: list[dict[str, Any]] = []
    for loader in DataLoader.CATALOG:
        d.append({
            "name": loader,
        })
    print(tabulate(d, headers="keys", tablefmt="pretty"))

@data.command
@click.option("--out", default="preprocessed.csv")
@click.option("--source-type", required=True)
@click.option("--source-name", required=True)
def preprocess(out: str, source_type: str, source_name: str):
    from src.data.sources.DataPuller import DataSource, DataPuller
    puller = DataPuller()

    print(f"Checking data source {source_type}/{source_name}...")
    print("(if it was already pulled then we will use it instead of pulling it again!)...")
    try:
        source = DataSource(source_type, source_name)
        puller.pull_data(source)
    except Exception as e:
        print(f"Failed to pull data: {e}")
        print("")
        return

    print("")
    start_time = time.monotonic()
    try:
        print()
        print("Running data processing pipeline...")
        from src.data.preprocessors.Preprocessor import Preprocessor
        from src.data.pipelines.PipelinesRunner import PipelinesRunner
        from src.data.pipelines.Pipeline import Pipeline
        from src.data.pipelines.Pipeline.PipelinePreviousStage import PipelinePreviousStage

        pipelines = [
            Pipeline("1", PipelinePreviousStage('loader', ['describing_timeseries']), [
                Preprocessor("autoencoder", {
                    "encoding_dim": "50",
                    "epochs": "50",
                    "batch_size": "32",
                }),
            ]),

            Pipeline("2", PipelinePreviousStage('loader', ['tabular']), [
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
        ]

        df = PipelinesRunner(pipelines, source, "7").run('train')[0]

        print(df)

        df.to_csv(out, index=False)

        return df
    finally:
        end_time = time.monotonic()
        print(f"Data Preprocessing Pipeline finished in {end_time - start_time} seconds")
