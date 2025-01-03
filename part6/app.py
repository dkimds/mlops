# app.py
import mlflow
import pandas as pd
from fastapi import FastAPI
from schemas import PredictIn, PredictOut


def get_model():
    model = mlflow.sklearn.load_model(model_uri="./sk_model")
    return model


MODEL = get_model()

# Create a FastAPI instance
app = FastAPI()


@app.post("/predict", response_model=PredictOut)
def predict(data: PredictIn) -> PredictOut:
    try:
        print(f"Received data: {data.dict()}")  # 요청 데이터 출력
        df = pd.DataFrame([data.dict()])
        pred = MODEL.predict(df).item()
        print(f"Prediction result: {pred}")  # 예측 결과 출력
        return PredictOut(iris_class=pred)
    except Exception as e:
        print(f"Error: {e}")  # 에러 발생 시 출력
        return {"error": str(e)}