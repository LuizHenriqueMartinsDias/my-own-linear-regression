import random
from pathlib import Path

import pandas as pd
from pandas import Series
from streamlit.watcher.path_watcher import get_default_path_watcher_class

BASE_DIR = Path(__file__).resolve().parent.parent
arquivo = BASE_DIR / "data" / "toy_dataset"
TOY_DATASET = pd.read_csv(arquivo)
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self,train_dataset):
        sumx = train_dataset["x"].sum()
        sumy = train_dataset["y"].sum()
        sumxy = (train_dataset["x"] * train_dataset["y"]).sum()
        sumx2 = (train_dataset["x"] ** 2).sum()
        n = len(train_dataset)

        self.coef_ = (n * sumxy - sumx * sumy) / (n * sumx2 - sumx ** 2)
        self.intercept_ = (sumy - self.coef_ * sumx) / n

    def predict(self,x):
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Please use the fit method before predicting")
        return self.coef_ * x + self.intercept_

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

class LinearRegressionGD:
    def __init__(self, learning_rate=0.001):
        self.coef_ = random.uniform(-1, 1)
        self.intercept_ = random.uniform(-1, 1)
        self.learning_rate = learning_rate

    def fit(self,train_dataset):
        for epoch in range(3500):
            predictions = []

            for x in train_dataset["x"]:
                predictions.append(self.coef_ * x + self.intercept_)
            predictions = pd.Series(predictions)
            mse = ((train_dataset["y"] - predictions) ** 2).mean()
            erros = []

            for index, y in enumerate(train_dataset["y"]):
                erros.append(y - predictions.iloc[index])

            erros = pd.Series(erros)

            erroxreal = (erros * train_dataset["x"]).sum()
            gd_w = (-2/len(train_dataset)) * erroxreal
            gd_y = (-2/len(train_dataset["y"]))*sum(erros)
            self.coef_ = self.coef_ - self.learning_rate * gd_w
            self.intercept_ = self.intercept_ - self.learning_rate * gd_y
    def predict(self,x):
        return self.coef_ * x + self.intercept_

def dataset_split(dataset,split:float):
    dt_split = len(dataset) * split
    validation_dataset = dataset.sample(
        n=int(dt_split),
        random_state=42
    )
    train_dataset = dataset.drop(validation_dataset.index)
    return train_dataset, validation_dataset

def main():
    train_dataset,validation_dataset=dataset_split(TOY_DATASET,0.2)
    lrgd = LinearRegressionGD()
    lrgd.fit(train_dataset)
    lr = LinearRegression()
    lr.fit(train_dataset)
    lrgd.fit(train_dataset)
    print(lrgd.predict(1),lr.predict(1))
if __name__ == "__main__":
    main()