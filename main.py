
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
    col_features: list
    city: str = "Probolinggo"
    days: int = 7

@app.get("/")
def root():
    return {"message": "API Prediksi Efisiensi Siap Jalan di Railway"}

@app.post("/predict")
def predict_rendemen(request: PredictRequest):
    try:
        # Ambil data cuaca (dengan fallback + caching)
        weather_df = fetch_weather_forecast(city=request.city, apikey=API_KEY, fallback_days=request.days)
        weather_df = weather_df.head(request.days)

        # Baca dataset
        df = pd.read_csv("prepare-2324.csv")

        # Latih model berdasarkan plant dan target
        model = train_model(df, request.plant_code, request.target_col, request.cuaca_features, request.col_features)

        # Prediksi dengan data cuaca
        X_pred = weather_df[request.cuaca_features]
        y_pred = model.predict(X_pred)

        # Masukkan hasil prediksi ke dataframe cuaca
        pred_col = 'prediksi_' + request.target_col
        weather_df[pred_col] = y_pred

        # Hitung statistik tambahan
        avg_pred = y_pred.mean()
        actual_mean = df[df['plant code'].str.upper() == request.plant_code.upper()][request.target_col].mean()
        percentage_prediction = round((avg_pred / actual_mean) * 100, 2) if actual_mean else None

        # Error estimasi (jika tersedia data aktual terakhir)
        if 'tanggal' in df.columns:
            df['tanggal'] = pd.to_datetime(df['tanggal'])
            recent = df.sort_values('tanggal', ascending=False).head(request.days)
            actuals = recent[request.target_col].values[:len(y_pred)]
            if len(actuals) == len(y_pred):
                mape = round((abs((actuals - y_pred) / actuals).mean()) * 100, 2)
            else:
                mape = None
        else:
            mape = None

        return {
            "summary": {
                "target": request.target_col,
                "plant_code": request.plant_code,
                "days": request.days,
                "mean_prediction": round(avg_pred, 3),
                "mean_actual": round(actual_mean, 3) if actual_mean else None,
                "prediction_percentage_vs_actual": percentage_prediction,
                "mape_error_estimate_percent": mape
            },
            "prediction": weather_df[['tanggal', pred_col]].to_dict(orient='records')
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#def predict_rendemen(request: PredictRequest):
#    try:
#        weather_df = fetch_weather_forecast(city=request.city, apikey=API_KEY, fallback_days=request.days)
#        weather_df = weather_df.head(request.days)

#        df = pd.read_csv("prepare-2324.csv")

#        model = train_model(df, request.plant_code, request.target_col, request.cuaca_features, request.col_features)

#        X_pred = weather_df[request.cuaca_features]
#        y_pred = model.predict(X_pred)

#        weather_df['prediksi_' + request.target_col] = y_pred
#        return weather_df[['tanggal', 'prediksi_' + request.target_col]].to_dict(orient='records')

#    except Exception as e:
#        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
