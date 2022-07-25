import pandas as pd
import numpy as np

path = "./PyTorchRefresher/data/train.csv/train.csv"

df = pd.read_csv(path, nrows=200000)

print(df.head())


def haversine_dist(df, lat1, long1, lat2, long2):
    r = 6371
    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])
    delta_phi = np.radians(df[lat2] - df[lat1])
    delta_lambda = np.radians(df[long2] - df[long1])
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = r * c
    return d


df["distance"] = haversine_dist(
    df, "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"
)
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df["edtDate"] = df["pickup_datetime"] - pd.Timedelta(hours=4)
df["hour"] = df["edtDate"].dt.hour
df["am"] = np.where(df.hour < 12, 1, 0)
df["weekday"] = df.edtDate.dt.strftime("%a")
print(df.head())
df.to_csv("./PyTorchRefresher/data/train.csv/train_processed.csv")
