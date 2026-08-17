import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DATASET = "toy_dataset"


class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, train_dataset):
        sumx = train_dataset["x"].sum()
        sumy = train_dataset["y"].sum()
        sumxy = (train_dataset["x"] * train_dataset["y"]).sum()
        sumx2 = (train_dataset["x"] ** 2).sum()
        n = len(train_dataset)

        self.coef_ = (n * sumxy - sumx * sumy) / (n * sumx2 - sumx ** 2)
        self.intercept_ = (sumy - self.coef_ * sumx) / n

    def predict(self, x):
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Please use the fit method before predicting")
        return self.coef_ * x + self.intercept_

    def score(self, validation_dataset):
        predictions = self.predict(validation_dataset["x"])
        sqr = ((predictions - validation_dataset["y"]) ** 2).sum()
        sqt = ((validation_dataset["y"].mean() - validation_dataset["y"]) ** 2).sum()
        print("R² =", 1 - (sqr / sqt))

    def plot_validation(self, train_dataset, validation_dataset):
        plt.scatter(train_dataset["x"], train_dataset["y"], label="Treino")
        plt.scatter(validation_dataset["x"], validation_dataset["y"], label="Validação")

        x = pd.concat([train_dataset["x"], validation_dataset["x"]])

        plt.plot(x, self.predict(x), label="Modelo")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()


class LinearRegressionGD:
    def __init__(self, learning_rate=0.0001, epochs=1200):
        self.coef_ = None
        self.intercept_ = None
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.mean_ = None
        self.std_ = None

    def fit(self, x, y):
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
            for _, row in weighted_x.iterrows():
                prediction = 0
                for elem in row:
                    prediction += elem
                predictions.append(prediction + self.intercept_)
            mse = (1 / n) * (((y - predictions) ** 2).sum())

            predictions = Series(predictions)
            error = predictions - y

            for index, column in enumerate(x):
                self.coef_[index] = self.coef_[index] - (
                    ((error * 2 * x[column]).sum() / n) * self.learning_rate
                )

            b_slope = (error * 2).sum() / n
            self.intercept_ = self.intercept_ - (self.learning_rate * b_slope)

            if epoch % 100 == 0:
                print("MSE:", mse)
                print("epoch:", epoch)
                print("w:", self.coef_)
                print("b:", self.intercept_)

    def predict(self, x: DataFrame | list):
        if isinstance(x, DataFrame):
            return x.to_numpy() @ np.asarray(self.coef_) + self.intercept_

        x = np.asarray(x, dtype=float)
        return np.asarray(self.coef_) @ x + self.intercept_

    def score(self, X, y):
        predictions = self.predict(X)
        sqr = ((predictions - y) ** 2).sum()
        sqt = ((y.mean() - y) ** 2).sum()
        print("R² =", 1 - (sqr / sqt))


def dataset_split(dataset, split: float):
    dt_split = len(dataset) * split
    validation_dataset = dataset.sample(n=int(dt_split), random_state=42)
    train_dataset = dataset.drop(validation_dataset.index)
    return (
        train_dataset.drop(columns="y"),
        train_dataset["y"],
        validation_dataset.drop(columns="y"),
        validation_dataset["y"],
    )


def standarization(x, mean=None, std=None):
    if mean is None:
        mean = x.mean(axis=0)
    if std is None:
        std = x.std(axis=0)

    x = (x - mean) / std
    return x, mean, std


def resolve_dataset_path(dataset: str) -> Path:
    candidate = Path(dataset)
    if candidate.is_file():
        return candidate

    in_data_dir = DATA_DIR / dataset
    if in_data_dir.is_file():
        return in_data_dir

    raise FileNotFoundError(
        f"Não encontrei o dataset '{dataset}'. Procurei em '{candidate}' e em '{in_data_dir}'. "
        f"Datasets disponíveis em {DATA_DIR}: {list_available_datasets()}"
    )


def list_available_datasets() -> list[str]:
    if not DATA_DIR.is_dir():
        return []
    return sorted(p.name for p in DATA_DIR.iterdir() if p.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina uma regressão linear (gradiente descendente) sobre um dataset CSV."
    )
    parser.add_argument(
        "-d", "--dataset",
        default=DEFAULT_DATASET,
        help=(
            "Nome do arquivo dentro de 'data/' ou caminho para um CSV. "
            f"(padrão: '{DEFAULT_DATASET}')"
        ),
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Lista os datasets disponíveis em 'data/' e sai.",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.4,
        help="Proporção do dataset usada para validação (padrão: 0.4).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Taxa de aprendizado do gradiente descendente (padrão: 0.1).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="Número de épocas de treinamento (padrão: 250).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_datasets:
        datasets = list_available_datasets()
        if datasets:
            print("Datasets disponíveis em", DATA_DIR)
            for name in datasets:
                print(" -", name)
        else:
            print("Nenhum dataset encontrado em", DATA_DIR)
        return

    dataset_path = resolve_dataset_path(args.dataset)
    print(f"Usando dataset: {dataset_path}")
    dataset = pd.read_csv(dataset_path)

    X_train, y_train, X_validation, y_validation = dataset_split(dataset, args.split)
    X_train, mean, std = standarization(X_train)
    X_validation, _, _ = standarization(X_validation, mean, std)

    lrgd = LinearRegressionGD(learning_rate=args.learning_rate, epochs=args.epochs)
    lrgd.fit(X_train, y_train)

    predictions = lrgd.predict(X_validation)
    print("Predições na validação:", predictions)
    lrgd.score(X_validation, y_validation)


if __name__ == "__main__":
    main()
