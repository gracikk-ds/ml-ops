# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utility import get_project_root


root = get_project_root()


@click.command()
@click.option("--input_filepath", default="data/raw/winequality-red.csv", type=str)
@click.option("--output_filepath", default="data/processed/", type=str)
def make_dataset(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    input_filepath = root / Path(input_filepath)
    output_filepath = root / Path(output_filepath)

    # Making binary classificaion for the response variable.
    # Dividing wine as good and bad by giving the limit for the quality
    bins = (2, 6.5, 8)
    group_names = ["bad", "good"]

    wine = pd.read_csv(input_filepath)
    wine["quality"] = pd.cut(wine["quality"], bins=bins, labels=group_names)

    # Now lets assign a labels to our quality variable
    label_quality = LabelEncoder()
    # Bad becomes 0 and good becomes 1
    wine["quality"] = label_quality.fit_transform(wine["quality"])

    # Now seperate the dataset as response variable and feature variabes
    X = wine.drop("quality", axis=1)
    y = wine["quality"]

    # Train and Test splitting of data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Applying Standard scaling to get optimized result
    sc = StandardScaler()

    X_train = pd.DataFrame(data=sc.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(data=sc.fit_transform(X_test), columns=X_test.columns)

    X_train["target"] = y_train.values
    X_test["target"] = y_test.values

    X_train.to_csv(output_filepath / "train.csv")
    X_test.to_csv(output_filepath / "test.csv")

    logger.info("done!")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    make_dataset()
