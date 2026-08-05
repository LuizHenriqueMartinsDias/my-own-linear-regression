from pathlib import Path

import pandas as pd
from pandas import Series

BASE_DIR = Path(__file__).resolve().parent.parent
arquivo = BASE_DIR / "data" / "toy_dataset"
TOY_DATASET = pd.read_csv(arquivo)

class LinearRegression:
    def __init__(self):
        self.b = None
        self.m = None

    def fit(self,train_dataset):
        sumx = train_dataset["x"].sum()
        sumy = train_dataset["y"].sum()
        sumxy = (train_dataset["x"] * train_dataset["y"]).sum()
        sumx2 = (train_dataset["x"] ** 2).sum()
        n = len(train_dataset)

        self.m = (n * sumxy - sumx * sumy) / (n * sumx2 - sumx ** 2)
        self.b = (sumy - self.m * sumx) / n

    def predict(self,x):
        return self.m * x + self.b

    def score(self,validation_dataset):
       predictions = self.predict(validation_dataset["x"])
       sqr = ((predictions - validation_dataset["y"])**2).sum()
       sqt = ((validation_dataset["y"].mean() - validation_dataset["y"])**2).sum()
       print("R² =",1-(sqr/sqt))

def main():
    train_dataset = TOY_DATASET[:25]
    validation_dataset = TOY_DATASET[25:]
    lr = LinearRegression()
    lr.fit(train_dataset)
    lr.predict(1)
    lr.score(validation_dataset)

if __name__ == "__main__":
    main()