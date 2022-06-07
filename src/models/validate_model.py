# -*- coding: utf-8 -*-
import json
import click
import pickle
import logging
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve


ROOT = Path(__file__).parent.parent.parent


@click.command()
@click.option("--path_to_dataset", default="data/processed/test.csv", type=str)
@click.option("--path_to_model_pkl", default="models/finalized_model.pkl", type=str)
@click.option("--path_to_metrics_storage", default="reports/metrics", type=str)
def main(path_to_dataset, path_to_model_pkl, path_to_metrics_storage):
    """Runs validation method"""

    logger = logging.getLogger(__name__)
    logger.info("Start predicting process")

    path_to_dataset = ROOT / Path(path_to_dataset)
    path_to_model_pkl = ROOT / Path(path_to_model_pkl)
    path_to_metrics_storage = ROOT / Path(path_to_metrics_storage)

    # read dataset
    test = pd.read_csv(path_to_dataset)

    # Now separate the dataset as response variable and feature variables
    x_test = test.drop("target", axis=1)
    y_test = test["target"]

    with open(path_to_model_pkl, "rb") as f:
        clf = pickle.load(f)

    predictions = clf.predict(x_test)
    predictions_proba = clf.predict_proba(x_test)

    # Let's see how our model performed
    precision = precision_score(y_test.values, predictions)
    recall = recall_score(y_test.values, predictions)
    roc_auc = roc_auc_score(y_test.values, predictions_proba[:, 1])
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
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
