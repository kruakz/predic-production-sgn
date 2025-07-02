
import requests
import pandas as pd
import os
import json
from datetime import datetime

DEFAULT_WEATHER = {
    "temp": 30.0,
    "humidity": 75,
    "pressure": 1010,
    "wind_speed": 2.0,
    "rain": 0.0
}

CACHE_DIR = "weather_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

in_memory_cache = {}

def make_cache_key(city):
    today = datetime.today().strftime("%Y-%m-%d")
    return (city.lower(), today)

def cache_filename(city):
    today = datetime.today().strftime("%Y-%m-%d")
    return os.path.join(CACHE_DIR, f"{city.lower()}_{today}.json")

def fetch_weather_forecast(city="Probolinggo", apikey="YOUR_API_KEY", fallback_days=7):
    key = make_cache_key(city)

    if key in in_memory_cache:
        return in_memory_cache[key]

    cache_file = cache_filename(city)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            in_memory_cache[key] = df
            return df
        except Exception:
            pass

    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?q={city},id&appid={apikey}&units=metric"
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        api_data = res.json()

        result = []
        for item in api_data["list"]:
            dt_txt = item["dt_txt"].split(" ")[0]
            result.append({
                "tanggal": dt_txt,
                "temp": item["main"]["temp"],
                "humidity": item["main"]["humidity"],
                "pressure": item["main"]["pressure"],
                "wind_speed": item["wind"]["speed"],
                "rain": item.get("rain", {}).get("3h", 0)
            })

        df = pd.DataFrame(result)
        df_grouped = df.groupby("tanggal").mean().reset_index()

        with open(cache_file, "w") as f:
            json.dump(df_grouped.to_dict(orient="records"), f)

        in_memory_cache[key] = df_grouped
        return df_grouped

    except Exception:
        fallback_data = [{
            "tanggal": (datetime.today().date() + pd.Timedelta(days=i)).strftime('%Y-%m-%d'),
            **DEFAULT_WEATHER
        } for i in range(fallback_days)]
        df = pd.DataFrame(fallback_data)
        in_memory_cache[key] = df
        return df
