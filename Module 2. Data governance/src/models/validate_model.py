# -*- coding: utf-8 -*-
import json
import click
import pickle
import logging
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report
from utility import get_project_root


root = get_project_root()


@click.command()
@click.option("--path_to_dataset", default="data/processed/test.csv", type=str)
@click.option("--path_to_model_pkl", default="models/finalized_model.pkl", type=str)
@click.option("--path_to_metrics_storage", default="reports/metrics", type=str)
def main(path_to_dataset, path_to_model_pkl, path_to_metrics_storage):
    """Runs validation method"""

    logger = logging.getLogger(__name__)
    logger.info("Start predicting process")

    path_to_dataset = root / Path(path_to_dataset)
    path_to_model_pkl = root / Path(path_to_model_pkl)
    path_to_metrics_storage = root / Path(path_to_metrics_storage)

    # read dataset
    test = pd.read_csv(path_to_dataset)

    # Now separate the dataset as response variable and feature variables
    x_test = test.drop("target", axis=1)
    y_test = test["target"]

    with open(path_to_model_pkl, "rb") as f:
        svc = pickle.load(f)

    predictions = svc.predict(x_test)

    # Let's see how our model performed
    metrics = classification_report(y_test, predictions, output_dict=True)

    metrics_per_class_0 = metrics["0"]
    metrics_per_class_1 = metrics["1"]
    accuracy = {"accuracy": metrics["accuracy"]}
    macro_avg = metrics["macro avg"]
    weighted_avg = metrics["weighted avg"]

    with open(
        str(path_to_metrics_storage / "metrics_per_class_0.json"), "w"
    ) as handler:
        json.dump(metrics_per_class_0, handler)
    with open(
        str(path_to_metrics_storage / "metrics_per_class_1.json"), "w"
    ) as handler:
        json.dump(metrics_per_class_1, handler)
    with open(str(path_to_metrics_storage / "accuracy.json"), "w") as handler:
        json.dump(accuracy, handler)
    with open(str(path_to_metrics_storage / "macro_avg.json"), "w") as handler:
        json.dump(macro_avg, handler)
    with open(str(path_to_metrics_storage / "weighted_avg.json"), "w") as handler:
        json.dump(weighted_avg, handler)

    logger.info("done!")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
