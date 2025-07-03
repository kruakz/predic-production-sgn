
import pandas as pd
from xgboost import XGBRegressor

def train_model(df, plant_code, target_col, cuaca_features,col_features):
    df = df[df['plant_code'].str.upper() == plant_code.upper()]
    if df.empty:
        raise ValueError("Data untuk plant_code tidak ditemukan.")

    all_cols = ['tgl_giling', target_col] + cuaca_features + col_features
    for col in all_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom {col} tidak ditemukan di dataset.")

    df = df[all_cols].dropna()
    df = df.rename(columns={'tgl_giling': 'tanggal', target_col: 'y'})
    df['tanggal'] = pd.to_datetime(df['tanggal'])

    X = df[cuaca_features]
    y = df['y']

    model = XGBRegressor()
    model.fit(X, y)

    return model
