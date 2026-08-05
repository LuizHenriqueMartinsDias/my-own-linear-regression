from pathlib import Path

import pandas as pd
from pandas import Series

BASE_DIR = Path(__file__).resolve().parent.parent
arquivo = BASE_DIR / "data" / "toy_dataset"
TOY_DATASET = pd.read_csv(arquivo)
import matplotlib.pyplot as plt

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

    def plot_validation(self, train_dataset, validation_dataset):
        plt.scatter(
            train_dataset["x"],
            train_dataset["y"],
            label="Treino"
        )

        plt.scatter(
            validation_dataset["x"],
            validation_dataset["y"],
            label="Validação"
        )

        x = pd.concat([
            train_dataset["x"],
            validation_dataset["x"]
        ])

        plt.plot(
            x,
            self.predict(x),
            label="Modelo"
        )

        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()
def dataset_split(dataset,split:float):
    dt_split = len(dataset) * split
    validation_dataset = dataset.sample(n=int(dt_split))
    train_dataset = dataset.drop(validation_dataset.index)
    return train_dataset, validation_dataset

def main():
    train_dataset, validation_dataset = dataset_split(TOY_DATASET, 0.2)
    lr = LinearRegression()
    lr.fit(train_dataset)
    lr.predict(1)
    lr.score(validation_dataset)
    lr.plot_validation(train_dataset,validation_dataset)
if __name__ == "__main__":
    main()