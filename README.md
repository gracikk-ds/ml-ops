## ML Engineering course by EPAM
The course includes 6 modules that cover the skills of a machine learning engineer.  

Skills of a machine learning engineer:
1. The ML Engineer should be proficient in all aspects of model architecture, data pipeline interaction, and metrics interpretation. 
2. A Machine Learning Engineer designs, builds, and productionizes ML models to solve business challenges.
3. The ML Engineer needs familiarity with foundational concepts of application development, infrastructure management, data engineering, and data governance. 
4. Through an understanding of training, retraining, deploying, scheduling, monitoring, and improving models, the ML Engineer designs and creates scalable solutions for optimal performance.


### Course modules:
-----
**Part 1 - Data Governance:**
* Make an initial setup using `DVC` tool and add a dataset :white_check_mark:
* Define a DVC pipeline that will:
  * preprocess data :white_check_mark:
  * train a model :white_check_mark:
  * evaluate the model :white_check_mark:
  * generate a feature importance plot with a model agnostic method :white_check_mark:
* The pipeline should be reproducible using `dvc repro` :white_check_mark:
* Run experiments and save metrics using `dvc metrics` :white_check_mark:
-----
**Part 2 - CI/CD and testing:**
* Create unit tests for python code from Part 1 :white_check_mark:
* Create a `github action` which at least performs:
    * code quality check :white_check_mark:
    * auto-formatting with black :white_check_mark:
    * linting with pylint - fail if less than a threshold example :white_check_mark:
    * run unit tests :white_check_mark:
-----
**Part 3 - Experiment tracking:**
* Conduct several experiments:
  * use different features :white_check_mark:
  * hyperparameter search :white_check_mark:
  * different models :white_check_mark:
* Results of each experiment should be tracked in MLFlow :white_check_mark:
* Best model artifacts should be logged as well :white_check_mark:
-----
**Part 4 - Model deployment in online mode (RestAPI):**
* Prepare model for deployment :white_check_mark:
* Use FastAPI for the deployment :white_check_mark:
* Wrap everything into Docker :white_check_mark: 
* Also prepare the service for cluster deployment :white_check_mark:
-----
**Part 5 - Python packaging:**
* Prepare python code for packaging :white_check_mark:
* Create and run tests :white_check_mark:
* Publish your package at pypi :white_check_mark:
-----
**Part 6 - ETL pipeline (link)[https://github.com/gracikk-ds/airflow/blob/main/dags/etl.py] :**
* Create an ETL pipeline as an Airflow dag :white_check_mark:
* Besides Pandas use Pyspark for data manipulation :white_check_mark:
-----
**Part 7 - Batch mode model deployment: **
* Create a pipeline for model training (link)[https://github.com/gracikk-ds/airflow/blob/main/dags/train.py]:
  * use the output from Part 6 as an input :white_check_mark:
  * define a logic for retraining :white_check_mark:
  * define a logic for model versioning :white_check_mark:
* Create a pipeline for model serving (link)[https://github.com/gracikk-ds/airflow/blob/main/dags/predict.py]:
  * use an output from Part 6 as an input :white_check_mark:
  * use a model from the model training pipeline as an input :white_check_mark:
-----

```bash
conda activate your_env
poetry config virtualenvs.path "path/to/your/conda/envs"
poetry config virtualenvs.create false
poetry install
```
