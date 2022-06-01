# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
from utility import get_project_root


root = get_project_root()


@click.command()
@click.option("--path_to_raw_data", type=str, default="data/raw/winequality-red.csv")
@click.option("--path_to_save_figs", type=str, default="reports/figures")
def main(path_to_raw_data: str, path_to_save_figs):
    logger = logging.getLogger(__name__)
    logger.info("raw data visualization stage")

    path_to_raw_data = root / Path(path_to_raw_data)
    path_to_save_figs = root / Path(path_to_save_figs)

    # Loading dataset
    wine = pd.read_csv(path_to_raw_data)

    # stage 1. fixed acidity
    logger.info("plotting fixed acidity barplot")
    plt.figure(figsize=(10, 6))
    sns.barplot(x="quality", y="fixed acidity", data=wine)
    plt.savefig(path_to_save_figs / "fixed_acidity.png")
    logger.info("done!")

    # stage 2. volatile acidity
    logger.info("plotting volatile acidity barplot")
    plt.figure(figsize=(10, 6))
    sns.barplot(x="quality", y="volatile acidity", data=wine)
    plt.savefig(path_to_save_figs / "volatile_acidity.png")
    logger.info("done!")

    # stage 3. citric acid
    logger.info("plotting citric acid barplot")
    plt.figure(figsize=(10, 6))
    sns.barplot(x="quality", y="citric acid", data=wine)
    plt.savefig(path_to_save_figs / "citric_acid.png")
    logger.info("done!")

    # stage 4. residual sugar
    logger.info("plotting residual sugar barplot")
    plt.figure(figsize=(10, 6))
    sns.barplot(x="quality", y="residual sugar", data=wine)
    plt.savefig(path_to_save_figs / "residual_sugar.png")
    logger.info("done!")

    # stage 5. chlorides
    logger.info("plotting chlorides barplot")
    plt.figure(figsize=(10, 6))
    sns.barplot(x="quality", y="chlorides", data=wine)
    plt.savefig(path_to_save_figs / "chlorides.png")
    logger.info("done!")

    # stage 6. free sulfur dioxide
    logger.info("plotting free sulfur dioxide barplot")
    plt.figure(figsize=(10, 6))
    sns.barplot(x="quality", y="free sulfur dioxide", data=wine)
    plt.savefig(path_to_save_figs / "free_sulfur_dioxide.png")
    logger.info("done!")

    # stage 7. total sulfur dioxide
    logger.info("plotting total sulfur dioxide barplot")
    plt.figure(figsize=(10, 6))
    sns.barplot(x="quality", y="total sulfur dioxide", data=wine)
    plt.savefig(path_to_save_figs / "total_sulfur_dioxide.png")
    logger.info("done!")

    # stage 8. sulphates
    logger.info("plotting sulphates barplot")
    plt.figure(figsize=(10, 6))
    sns.barplot(x="quality", y="sulphates", data=wine)
    plt.savefig(path_to_save_figs / "sulphates.png")
    logger.info("done!")

    # stage 9. alcohol
    logger.info("plotting alcohol barplot")
    plt.figure(figsize=(10, 6))
    sns.barplot(x="quality", y="alcohol", data=wine)
    plt.savefig(path_to_save_figs / "alcohol.png")
    logger.info("done!")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
