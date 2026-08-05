from pathlib import Path

import pandas as pd
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
        y = self.m * x + self.b
        print(y)

def main():
    lr = LinearRegression()
    lr.fit(TOY_DATASET)
    lr.predict(1)

if __name__ == "__main__":
    main()