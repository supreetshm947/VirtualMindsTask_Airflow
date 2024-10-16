from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/predict")
async def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = model.predict(features)

    return {"prediction": prediction[0]}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Variety Prediction API!"}