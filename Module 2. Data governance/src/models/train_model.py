# -*- coding: utf-8 -*-
import click
import pickle
import logging
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


@click.command()
@click.option("--path_to_dataset", default="../../data/processed/train.csv", type=str)
@click.option("--path_to_model_storage", default="../../models", type=str)
def main(path_to_dataset, path_to_model_storage):
    """Runs training job and save best model to model's storage"""

    logger = logging.getLogger(__name__)
    logger.info("Start model training process")

    path_to_dataset = Path(path_to_dataset)
    path_to_model_storage = Path(path_to_model_storage)

    # read dataset
    train = pd.read_csv(path_to_dataset)

    # Now separate the dataset as response variable and feature variables
    x_train = train.drop("target", axis=1)
    y_train = train["target"]

    # init svc model
    svc = SVC()

    # Finding best parameters for SVC model
    param = {"C": [0.8, 1, 1.2], "kernel": ["linear", "rbf"], "gamma": [0.8, 1, 1.2]}
    grid_svc = GridSearchCV(svc, param_grid=param, scoring="roc_auc", cv=3)

    grid_svc.fit(x_train, y_train)

    # Let's run SVC again with the best parameters.
    svc = SVC(**grid_svc.best_params_)
    svc.fit(x_train, y_train)

    with open(str(path_to_model_storage / "finalized_model.pkl"), "wb") as handle:
        pickle.dump(svc, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("done!")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
