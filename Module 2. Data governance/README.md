Data Governance, Experiment tracking and Model Deployment Project
==============================
The tasks of the project could be divided into 4 main parts:

**Part 1 - Data Governance:**
* Make an initial setup using `Data version control` tool and add a dataset, so another could obtain it via `dvc pull`
* Define a DVC pipeline that will:
  * preprocess data
  * train a model
  * evaluate the model
  * generate a feature importance plot with a model agnostic method
  The pipeline should be reproducible using `dvc repro`
* Run experiments and save metrics using `dvc metrics`

**Part 2 - CICD, testing:**
* Create unit tests for python code from Part 1
* Create a `github action` which at least performs:
    * code quality check:
    * auto-formatting with black
    * linting with pylint - fail if less than a threshold example
    * run unit tests

**Part 3 -  Experiment tracking:**
* Conduct several experiments:
  * use different features
  * hyperparameter search
  * different models
* Results of each experiment should be tracked in MLFlow
* Best model artifacts should be logged as well


Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io



Conda + Poetry usage
--------
```bash
# install poetry macos
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

# activate conda env and install dependencies
conda activate your_env
poetry config virtualenvs.path "path/to/your/conda/envs"
poetry config virtualenvs.create false
poetry install
```
