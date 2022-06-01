# -*- coding: utf-8 -*-
import shap
import yaml
import json
import click
import pickle
import optuna
import logging
import pandas as pd
from pathlib import Path
from functools import partial
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from utility import get_project_root


root = get_project_root()


def objective(trial, params, x_train, y_train):
    penalty = trial.suggest_categorical("penalty", params["penalty"])
    c = trial.suggest_float(
        "C",
        float(params["C"]["upper_bound"]),
        float(params["C"]["lower_bound"]),
        log=True,
    )

    clf = LogisticRegression(solver="liblinear", penalty=penalty, C=c)

    target = cross_val_score(
        clf, x_train, y_train, n_jobs=-1, cv=3, scoring="precision"
    ).mean()

    return target


@click.command()
@click.option("--path_to_dataset", default="data/processed/train.csv", type=str)
@click.option("--path_to_model_storage", default="models", type=str)
@click.option("--path_to_metrics_storage", default="reports/metrics", type=str)
def main(path_to_dataset, path_to_model_storage, path_to_metrics_storage):
    """Runs training job and save best model to model's storage"""

    logger = logging.getLogger(__name__)
    logger.info("Start model training process")

    path_to_dataset = root / Path(path_to_dataset)
    path_to_model_storage = root / Path(path_to_model_storage)
    path_to_metrics_storage = root / Path(path_to_metrics_storage)

    # read dataset
    train = pd.read_csv(path_to_dataset)

    # Now separate the dataset as response variable and feature variables
    x_train = train.drop("target", axis=1)
    y_train = train["target"]

    # save best params
    with open(root / Path("params.yaml"), "r") as stream:
        params = yaml.safe_load(stream)["train"]
        print(params)

    obj_partial = partial(objective, params=params, x_train=x_train, y_train=y_train)

    study = optuna.create_study(direction="maximize")
    study.optimize(obj_partial, n_trials=200)

    trial = study.best_trial

    print("Best hyperparameters: {}".format(trial.params))

    # Let's run SVC again with the best parameters.
    clf = LogisticRegression(solver="liblinear", **trial.params)
    clf.fit(x_train, y_train)

    explainer = shap.Explainer(clf.predict, x_train)
    shap_values = explainer(x_train)

    # summarize the effects of all the features
    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig(
        root / "reports/features_importance/shap_values.png",
        format="png",
        dpi=150,
        bbox_inches="tight",
    )

    with open(str(path_to_metrics_storage / "hyper_params.json"), "w") as handler:
        json.dump(trial.params, handler)

    with open(str(path_to_model_storage / "finalized_model.pkl"), "wb") as handle:
        pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("done!")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
