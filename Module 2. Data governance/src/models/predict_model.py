# -*- coding: utf-8 -*-
import click
import pickle
import logging
import datetime
import pandas as pd
from pathlib import Path
from src.utils import get_project_root


root = get_project_root()


@click.command()
@click.option("--path_to_dataset", default="data/processed/test.csv", type=str)
@click.option("--path_to_model_pkl", default="models/finalized_model.pkl", type=str)
@click.option("--path_to_predictions_storage", default="data/predictions", type=str)
def main(path_to_dataset, path_to_model_pkl, path_to_predictions_storage):
    """Runs model's predict method"""

    logger = logging.getLogger(__name__)
    logger.info("Start predicting process")

    path_to_dataset = root / Path(path_to_dataset)
    path_to_model_pkl = root / Path(path_to_model_pkl)
    path_to_predictions_storage = root / Path(path_to_predictions_storage)

    # read dataset
    x_test = pd.read_csv(path_to_dataset)

    # Now separate the dataset as response variable and feature variables
    if "target" in x_test.columns:
        x_test = x_test.drop("target", axis=1)

    with open(path_to_model_pkl, "rb") as f:
        svc = pickle.load(f)

    predictions = pd.DataFrame(data=svc.predict(x_test), columns=["wine_quality"])

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    predictions.to_csv(
        path_to_predictions_storage / ("predictions_" + current_time + ".csv")
    )

    logger.info("done!")


# if __name__ == "__main__":
#     log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#     logging.basicConfig(level=logging.INFO, format=log_fmt)
#     main()
