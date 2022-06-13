# -*- coding: utf-8 -*-
import os
import shap
import yaml
import json
import click
import mlflow
import optuna
import logging
import pandas as pd
from pathlib import Path
from functools import partial
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from mlflow.models import infer_signature
from mlflow.tracking.client import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


load_dotenv()
remote_server_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(remote_server_uri)

ROOT = Path(__file__).parent.parent.parent


def objective(trial, params, x_train, y_train):
    """Run optuna trials"""
    with mlflow.start_run(nested=True):
        penalty = trial.suggest_categorical("penalty", params["penalty"])
        c = trial.suggest_float(
            "C",
            float(params["C"]["upper_bound"]),
            float(params["C"]["lower_bound"]),
            log=True,
        )

        clf = LogisticRegression(solver="liblinear", penalty=penalty, C=c)

        mlflow.log_params({"C": c, "penalty": penalty})

        target = cross_val_score(
            clf, x_train, y_train, n_jobs=-1, cv=3, scoring="precision"
        ).mean()

        mlflow.log_metrics({"train_precision": target})

    return target


@click.command()
@click.option("--path_to_dataset", default="data/processed/train.csv", type=str)
def main(path_to_dataset):
    mlflow.set_experiment("linear_model_experiments_1")

    with mlflow.start_run() as run:
        """Runs training job and save best model to model's storage"""

        experiment_id = run.info.experiment_id

        logger = logging.getLogger(__name__)
        logger.info("Start model training process")

        path_to_dataset = ROOT / Path(path_to_dataset)

        # read dataset
        train = pd.read_csv(path_to_dataset).drop(columns=["Unnamed: 0"])

        # Now separate the dataset as response variable and feature variables
        x_train = train.drop("target", axis=1)

        print(x_train.head())
        y_train = train["target"]

        # save best params
        with open(ROOT / Path("params.yaml"), "r") as stream:
            params = yaml.safe_load(stream)["train"]

        obj_partial = partial(objective, params=params, x_train=x_train, y_train=y_train)

        study = optuna.create_study(direction="maximize")
        study.optimize(obj_partial, n_trials=200)

        # find the best run, log its metrics as the final metrics of this run.
        client = MlflowClient()
        runs = client.search_runs(
            [experiment_id],
            "tags.mlflow.parentRunId = '{run_id}' ".format(run_id=run.info.run_id)
        )

        best_metric = 0
        best_run = None
        for r in runs:
            if r.data.metrics["train_precision"] > best_metric:
                if best_run is not None:
                    run_id = best_run.info.run_id
                    mlflow.delete_run(run_id)
                best_run = r
                best_metric = r.data.metrics["train_precision"]
            else:
                run_id = r.info.run_id
                mlflow.delete_run(run_id)
        mlflow.set_tag("params_search_best_run", best_run.info.run_id)

        mlflow.log_metrics(
            {
                "train_precision": best_metric,
            }
        )

        mlflow.log_params(
            {
                "C": best_run.data.params["C"],
                "penalty": best_run.data.params["penalty"]
            }
        )

        # Let's run SVC again with the best parameters.
        clf = LogisticRegression(solver="liblinear", **study.best_trial.params)
        clf.fit(x_train, y_train)

        signature = infer_signature(x_train, y_train)

        # Log the sklearn model and register as version 1
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="sklearn-model",
            registered_model_name="sk-learn-log-reg",
            signature=signature,
        )

        explainer = shap.Explainer(clf.predict, x_train)
        shap_values = explainer(x_train)

        # summarize the effects of all the features
        shap.plots.beeswarm(shap_values, show=False)
        plt.savefig(
            ROOT / "reports/features_importance/shap_values.png",
            format="png",
            dpi=150,
            bbox_inches="tight",
        )

        logger.info("done!")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
