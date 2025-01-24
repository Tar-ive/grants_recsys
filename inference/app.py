from fastapi import FastAPI
from predict import predict

app = FastAPI()

@app.post("/predict")
async def predict_endpoint(input_data: dict):
    return {"prediction": predict(input_data)}