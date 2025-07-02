
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import train_model
from weather import fetch_weather_forecast
import pandas as pd
import os
import uvicorn

app = FastAPI()

API_KEY = os.getenv("YOUR_API_KEY", "f76a546fc63ea2eb0bec6bbd196c0521")

class PredictRequest(BaseModel):
    plant_code: str
    target_col: str
    cuaca_features: list
    city: str = "Probolinggo"
    days: int = 7

@app.get("/")
def root():
    return {"message": "API Prediksi Efisiensi Siap Jalan di Railway"}

@app.post("/predict")
def predict_rendemen(request: PredictRequest):
    try:
        weather_df = fetch_weather_forecast(city=request.city, apikey=API_KEY, fallback_days=request.days)
        weather_df = weather_df.head(request.days)

        df = pd.read_csv("prepare-dataset.csv")

        model = train_model(df, request.plant_code, request.target_col, request.cuaca_features)

        X_pred = weather_df[request.cuaca_features]
        y_pred = model.predict(X_pred)

        weather_df['prediksi_' + request.target_col] = y_pred
        return weather_df[['tanggal', 'prediksi_' + request.target_col]].to_dict(orient='records')

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
