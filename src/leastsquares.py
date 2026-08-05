from pathlib import Path

import pandas as pd
BASE_DIR = Path(__file__).resolve().parent.parent
arquivo = BASE_DIR / "data" / "toy_dataset.csv"
TOY_DATASET = pd.read_csv(arquivo)

class LinearRegression:
    def __init__(self,train_dataset,val_dataset):
        self.b = None
        self.m = None
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def get_param(self):
        sumx = sum(self.train_dataset["x"].values)
        sumy = sum(self.train_dataset["y"].values)
        sumxy = sum(self.train_dataset["x"].values * self.train_dataset["y"].values)
        sumxs = sum(self.train_dataset["x"] ** 2)
        n = len(self.train_dataset["x"])
        self.m = n * sumxy - sumx * sumy/n * sumxs - (sumx**2)
        self.b = sumy - self.m * sumx/n

    def linear_function(self,x):
        self.get_param()
        y = self.m * x + self.b
        print(y)

def main():
    lr = LinearRegression(TOY_DATASET,TOY_DATASET)
    lr.linear_function(1)

if __name__ == "__main__":
    main()