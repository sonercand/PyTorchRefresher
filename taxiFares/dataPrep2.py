import pandas as pd
import numpy as np
import torch

data = pd.read_csv("./PyTorchRefresher/data/train.csv/train_processed.csv")
print(data.head())
print(data.shape)

data["distance"] = data["distance"].fillna(data["distance"].mean())
catCols = ["hour", "am", "weekday"]
contCols = ["distance", "passenger_count"]
label = ["fare_amount"]
for cat in catCols:
    data[cat] = data[cat].astype("category")
print(data.dtypes)
# convert cat values
hr = data["hour"].cat.codes.values
am = data["am"].cat.codes.values
wkd = data["weekday"].cat.codes.values
cats = np.stack([hr, am, wkd], axis=1)
print(cats)
cats = torch.tensor(cats, dtype=torch.int64)
# convert cont. values

conts = np.stack([data[col].values for col in contCols], axis=1)
conts = torch.tensor(conts, dtype=torch.float)
# convert label
label = torch.tensor(data[label].values, dtype=torch.float).reshape(-1, 1)
print(cats.shape, conts.shape, label.shape)
torch.save(cats, "./PyTorchRefresher/data/train.csv/cats.pt")
torch.save(conts, "./PyTorchRefresher/data/train.csv/conts.pt")
torch.save(label, "./PyTorchRefresher/data/train.csv/label.pt")
