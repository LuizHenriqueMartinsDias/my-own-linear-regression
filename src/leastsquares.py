import random
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

BASE_DIR = Path(__file__).resolve().parent.parent
arquivo = BASE_DIR / "data" / "toy_dataset"
TOY_DATASET = pd.read_csv(arquivo)

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

class LinearRegressionGD:
    def __init__(self, learning_rate=0.0001,epochs=1200):
        super().__init__()
        self.coef_ = None
        self.intercept_ = None
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, x,y):
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)
        self.intercept_ = 0.0
        self.coef_ = [0.0 for _ in x.columns]
        n = len(x)

        for epoch in range(self.epochs):
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
            if epoch % 10 == 0:
                print("MSE:",mse)
                print("epoch:",epoch)
                print("w:",self.coef_)
                print("b:",self.intercept_)

    def predict(self, x: DataFrame | list):
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Please use the fit method before predicting")
        if isinstance(x, DataFrame):
            return x.to_numpy() @ np.asarray(self.coef_) + self.intercept_

        x = np.asarray(x, dtype=float)
        return np.asarray(self.coef_) @ x + self.intercept_

    def score(self,X,y):
       predictions = self.predict(X)
       sqr = ((predictions - y)**2).sum()
       sqt = ((y.mean() - y)**2).sum()
       print("R² =",1-(sqr/sqt))

def dataset_split(dataset,split:float):

    dt_split = len(dataset) * split
    validation_dataset = dataset.sample(
        n=int(dt_split),
        random_state=42
    )
    train_dataset = dataset.drop(validation_dataset.index)
    return train_dataset.drop(columns="y"),train_dataset["y"], validation_dataset.drop(columns="y"),validation_dataset["y"]
def standarization(x, mean=None, std=None):
    if mean is None:
        mean = x.mean(axis=0)

    if std is None:
        std = x.std(axis=0)

    x = (x - mean) / std

    return x, mean, std

def main():
    X_train,y_train,X_validation,y_validation = dataset_split(TOY_DATASET,0.4)
    X_train, mean, std = standarization(X_train)

    X_validation, _, _ = standarization(
        X_validation,
        mean,
        std
    )
    lrgd = LinearRegressionGD(learning_rate=0.1,epochs=250)
    lrgd.fit(X_train,y_train)
    x_novo = pd.DataFrame([[45, 1, 18]], columns=["area_m2", "quartos", "idade_anos"])
    x_novo_std, _, _ = standarization(x_novo, mean, std)
    print(lrgd.predict(X_validation))
    lrgd.score(X_validation,y_validation)

if __name__ == "__main__":
    main()
