# -*- coding: utf-8 -*-
import os
import json
import click
import mlflow
import logging
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from mlflow.tracking.client import MlflowClient
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve


@click.command()
@click.option("--path_to_dataset", default="data/processed/test.csv", type=str)
@click.option("--path_to_metrics_storage", default="reports/metrics", type=str)
@click.option("--registered_model_name", default="default_model", type=str)
@click.option("--experiment_name", default="default_experiment")
def main(
    path_to_dataset, path_to_metrics_storage, registered_model_name, experiment_name
):
    """Runs validation method"""
    client = MlflowClient()

    if experiment_name is None:

        experiments = client.list_experiments()
        current_experiment = experiments[-1]
        df = mlflow.search_runs([current_experiment.experiment_id])
        df.sort_values(by="start_time", inplace=True)
        df.reset_index(inplace=True, drop=True)
        run_id = df.run_id.values[-1]

    else:
        current_experiment = mlflow.get_experiment_by_name(experiment_name)
        df = mlflow.search_runs([current_experiment.experiment_id])
        df.sort_values(by="start_time", inplace=True)
        df.reset_index(inplace=True, drop=True)
        run_id = df.run_id.values[-1]

    with mlflow.start_run(run_id=run_id):
        logger = logging.getLogger(__name__)
        logger.info("Start predicting process")

        path_to_dataset = ROOT / Path(path_to_dataset)
        path_to_metrics_storage = ROOT / Path(path_to_metrics_storage)

        # read dataset
        test = pd.read_csv(path_to_dataset).drop(columns=["Unnamed: 0"])

        # Now separate the dataset as response variable and feature variables
        x_test = test.drop("target", axis=1)
        y_test = test["target"]

        latest_version = client.get_latest_versions(
            registered_model_name, stages=["None"]
        )[0].version

        clf = mlflow.sklearn.load_model(
            f"models:/{registered_model_name}/{latest_version}"
        )

        predictions = clf.predict(x_test)
        predictions_proba = clf.predict_proba(x_test)

        # Let's see how our model performed
        precision = precision_score(y_test.values, predictions)
        recall = recall_score(y_test.values, predictions)
        roc_auc = roc_auc_score(y_test.values, predictions_proba[:, 1])

        mlflow.log_metrics({"test_precision": precision})
        mlflow.log_metrics({"test_recall": recall})
        mlflow.log_metrics({"test_roc_auc": roc_auc})

        fpr, tpr, _ = roc_curve(y_test.values, predictions_proba[:, 1])

        metrics = {
            "train": {
                "precision": precision,
                "recall": recall,
                "roc_auc": roc_auc,
            }
        }

        plots = {"train": [{"tpr": i, "fpr": j} for i, j in zip(tpr, fpr)]}

        with open(str(path_to_metrics_storage / "metrics.json"), "w") as handler:
            json.dump(metrics, handler)

        with open(str(path_to_metrics_storage / "plots.json"), "w") as handler:
            json.dump(plots, handler)

        logger.info("done!")


if __name__ == "__main__":
    load_dotenv()
    remote_server_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(remote_server_uri)

    ROOT = Path(__file__).parent.parent.parent

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
