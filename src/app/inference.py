import os
import mlflow
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException

# load env variables
load_dotenv(find_dotenv())

# Initialize FastApi application
app = FastAPI()


class Model(object):
    def __init__(self, registered_model_name, model_stage):
        """
        To initialize the model
        registered_model_name: name of the model in registry
        model_stage: stage of the model
        """

        # load the model
        self.model = mlflow.sklearn.load_model(
            f"models:/{registered_model_name}/{model_stage}"
        )

    def inference(self, data):
        """
        Make prediction using loaded model
        data: pd.DataFrame to perform prediction
        """
        predictions = self.model.predict(data)
        print("hi!")
        return predictions


model = Model("default_model", "Staging")


# welcome message.
@app.get("/")
async def root():
    return {
        "message": "Hello, Dear User! "
        "You could download file to inference using /invocations endpoint"
    }


# Create POST endpoint with path /invocations
@app.post("/invocations")
async def upload_file(file: UploadFile = File(...)):
    print(file)
    if file.filename.endswith(".csv"):
        with open(file.filename, "wb") as f:
            f.write(file.file.read())
        data = pd.read_csv(file.filename)
        os.remove(file.filename)

        data = data.loc[
            :, [x for x in list(data.columns) if x not in ["Unnamed: 0", "target"]]
        ]

        predictions = model.inference(data)

        return [int(x) for x in predictions]
    else:
        # Raise HTTP 400 exeption
        HTTPException(status_code=400, detail="Invalid file type. Only .csv accepted")
