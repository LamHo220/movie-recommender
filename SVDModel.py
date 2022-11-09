# SVD
## Import libraries
from lib.models import RecommendSystemModel

from typing import List, Any, Tuple, Union
from numpy.typing import NDArray
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# The ML model
class SVDModel(RecommendSystemModel):
    def __init__(
        self,
        mode: str = None,
        features: int = None,
        lr: float = None,
        epochs: int = None,
        weight_decay: float = None,
        stopping: float = None,
        momentum: float = None,
    ) -> None:
        super().__init__(mode, features, lr, epochs, weight_decay, stopping, momentum)

    def split(
        self, ratio_train_test: float, ratio_train_valid: float, tensor: bool = False
    ) -> None:
        userItemMatrix = self.data[['userId','movieId','rating']].pivot_table(columns='movieId', index='userId', values='rating').fillna(0)
        
        n = len(userItemMatrix)
        m = len(userItemMatrix.columns)

        trainBeforeSplit = userItemMatrix.copy().apply(lambda x: pd.Series(map(lambda y:0,x)))
        self.train = trainBeforeSplit.copy()
        self.valid = trainBeforeSplit.copy()
        self.test = trainBeforeSplit.copy()

        for i in range(n):
            for j in range(m):
                if userItemMatrix.iloc[i,j]:
                    if np.random.binomial(1, ratio_train_test, 1):
                        trainBeforeSplit.iloc[i,j] = userItemMatrix.iloc[i,j]
                    else:
                        self.test.iloc[i,j] = userItemMatrix.iloc[i,j]

        for i in range(n):
            for j in range(m):
                if trainBeforeSplit.iloc[i,j]:
                    if np.random.binomial(1, ratio_train_valid, 1):
                        self.train.iloc[i,j] = trainBeforeSplit.iloc[i,j]
                    else:
                        self.valid.iloc[i,j] = trainBeforeSplit.iloc[i,j]

    def data_loader(
        self,
        path: str = None,
        nrows: int = None,
        skiprows=None,
        data: pd.DataFrame = None,
        n_users: int = None,
        n_items=None,
    ) -> None:
        if not path and data.empty:
            raise "Error: one of path or data frame should be provided"
        if data.empty:
            self.data = pd.read_csv(
                path, low_memory=False, nrows=nrows, skiprows=skiprows
            )
        elif not path:
            self.data = data
        self.users = self.data['userId'].unique()
        self.movies = self.data['movieId'].unique()
        self.n_users = len(self.users)
        self.n_items = len(self.movies)

    def training(self) -> Tuple[NDArray, NDArray, float, float]:
        loss_train = []
        loss_valid = []
        errors = []

        self._P = np.random.rand(self.n_users, self.features) * 0.1
        self._Q = np.random.rand(self.n_items, self.features) * 0.1

        # for advanced SVD
        self._bu = np.zeros(self.n_users)
        self._bi = np.zeros(self.n_items)
        self.mean = 0  # TODO calculate the mean of rating

        # Johnny
        for e in range(self.epochs):
            for id_user, user in enumerate(self.users):
                for id_item, movie in enumerate(self.movies):
                    if self.train.iloc[user, movie] > 0:

                        predict = self.prediction(id_user, id_item)

                        error = self.train.iloc[user, movie] - predict
                        errors.append(error)

                        self.optimize(error, id_user, id_item)
            trainLoss = self.loss(self.train)
            validLoss = self.loss(self.valid)
            loss_train.append(trainLoss)
            loss_valid.append(validLoss)
            if e % 10 == 0:
                print(
                    "Epoch : ",
                    "{:3.0f}".format(e + 1),
                    " | Train :",
                    "{:3.3f}".format(trainLoss),
                    " | Valid :",
                    "{:3.3f}".format(validLoss),
                )

            if e > 1:
                if abs(validLoss - trainLoss) < self.stopping:
                    break
        print("Training stopped:")
        print(
            "Epoch : ",
            "{:3.0f}".format(e + 1),
            " | Train :",
            "{:3.3f}".format(trainLoss),
            " | Valid :",
            "{:3.3f}".format(validLoss),
        )
        return loss_train, loss_valid, errors


    def prediction(self, u: int, i: int) -> float:
        # Woody
        predict = np.dot(self._P[u, :], self._Q[i, :])
        if self.mode == "svd++":
            predict += self._mean + self._bu[u] + self._bi[i]
        return predict

    def loss(self, groundTruthData) -> float:
        # Woody
        squaredErrors = 0.0
        numOfPrediction = 0
        for u,user in enumerate(self.users):
            for i,movie in enumerate(self.movies):
                if groundTruthData.iloc[user,movie] > 0:
                    squaredErrors += pow(
                        groundTruthData.iloc[user,movie] - self.prediction(u, i), 2
                    )
                    numOfPrediction += 1
        return squaredErrors / numOfPrediction

    def optimize(self, error: float, id_user: int, id_item: int):
        # Johnny
        self._P[id_user, :] += self.lr * (
            error * self._Q[id_item, :] - self.weight_decay * self._P[id_user, :]
        )
        self._Q[id_item, :] += self.lr * (
            error * self._P[id_user, :] - self.weight_decay * self._Q[id_item, :]
        )
