import random
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

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

class LinearRegressionGD():
    def __init__(self, learning_rate=0.0001):
        super().__init__()
        self.coef_ = None
        self.intercept_ = None
        self.learning_rate = learning_rate

    def fit(self, train_dataset):
        x = train_dataset.drop(columns="y")
        y = train_dataset["y"]
        self.intercept_ = random.uniform(-1, 1)
        self.coef_ = [random.uniform(-1, 1) for _ in x.columns]
        n = len(train_dataset)

        for epoch in range(1200):
            predictions = []
            weighted_x = x.copy()
            for index, c in enumerate(x):
                weighted_x[c] = weighted_x[c] * self.coef_[index]
            for index, row in weighted_x.iterrows():
                prediction = 0
                for elem in row:
                    prediction += elem
                predictions.append(prediction + self.intercept_)
            mse = (1/n)*(((y - predictions)**2).sum())

            predictions = Series(predictions)
            error = predictions - y

            for index, column in enumerate(x):
                self.coef_[index] = self.coef_[index] - (((error * 2 * x[column]).sum() / n) * self.learning_rate)

            b_slope = (error * 2).sum() / n
            self.intercept_ = self.intercept_ - (self.learning_rate * b_slope)
            print("MSE:",mse)
            print("epoch:",epoch)
            print("w:",self.coef_)
            print("b:",self.intercept_)

    def predict(self,x:DataFrame|list):
        if isinstance(x, DataFrame):
            return [np.ndarray(self.coef_) @ x] + self.intercept_
        return sum([x * y for x, y in zip(self.coef_, x)] ) + self.intercept_

def dataset_split(dataset,split:float):
    dt_split = len(dataset) * split
    validation_dataset = dataset.sample(
        n=int(dt_split),
        random_state=42
    )
    train_dataset = dataset.drop(validation_dataset.index)
    return train_dataset, validation_dataset

def main():
    train_dataset,validation_dataset = dataset_split(TOY_DATASET,0.4)
    lrgd = LinearRegressionGD(learning_rate=0.0001)
    lrgd.fit(train_dataset)
    print(lrgd.predict([1,10,5]))

if __name__ == "__main__":
    main()