from lib.models import RecommendSystemModel

from typing import List, Any, Tuple, Union
from numpy.typing import NDArray
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# import tensorflow as tf
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
    ) -> List[NDArray]:
        userItemMatrix = self._convertToUserItemMatrix(
            self.data, self.n_users, self.n_items
        )

        trainBeforeSplit = np.zeros(
            (len(userItemMatrix), len(userItemMatrix[0]))
        ).tolist()
        self.train = np.zeros((len(userItemMatrix), len(userItemMatrix[0]))).tolist()
        self.valid = np.zeros((len(userItemMatrix), len(userItemMatrix[0]))).tolist()
        self.test = np.zeros((len(userItemMatrix), len(userItemMatrix[0]))).tolist()

        for i in range(len(userItemMatrix)):
            for j in range(len(userItemMatrix[i])):
                if userItemMatrix[i][j] > 0:
                    if np.random.binomial(1, ratio_train_test, 1):
                        trainBeforeSplit[i][j] = userItemMatrix[i][j]
                    else:
                        self.test[i][j] = userItemMatrix[i][j]

        for i in range(len(trainBeforeSplit)):
            for j in range(len(trainBeforeSplit[i])):
                if trainBeforeSplit[i][j] > 0:
                    if np.random.binomial(1, ratio_train_valid, 1):
                        self.train[i][j] = trainBeforeSplit[i][j]
                    else:
                        self.valid[i][j] = trainBeforeSplit[i][j]

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
        self.n_users = n_users
        self.n_items = n_items

    def training(self) -> Tuple[NDArray, NDArray, float, float]:
        loss_train = []
        loss_valid = []
        errors = []

        self._P = np.random.rand(self.n_users, self.features) * 0.1
        self._Q = np.random.rand(self.n_items, self.features) * 0.1

        self._bu = np.zeros(self.n_users)
        self._bi = np.zeros(self.n_items)
        self.mean = 0  # TODO calculate the mean of rating

        # Johnny
        for e in range(self.epochs):
            for id_user in range(self.n_users):
                for id_item in range(self.n_items):
                    if self.train[id_user][id_item] > 0:

                        predict = self.prediction(self._P, self._Q, id_user, id_item)

                        error = self.train[id_user][id_item] - predict
                        errors.append(error)

                        self.optimize(error, id_user, id_item, self.weight_decay)
            trainLoss = self.loss(self.train, self._P, self._Q)
            validLoss = self.loss(self.valid, self._P, self._Q)
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
                if validLoss - loss_valid[-2] < self.stopping:
                    print("Training stopped:")
                    print(
                        "Epoch : ",
                        "{:3.0f}".format(e + 1),
                        " | Train :",
                        "{:3.3f}".format(trainLoss),
                        " | Valid :",
                        "{:3.3f}".format(validLoss),
                    )
                    break
        return loss_train, loss_valid, errors

    def _convertToUserItemMatrix(self, data, n_users, n_movies):
        data = np.array(data, dtype=int)
        userItemMatrix = []
        for id_user in range(1, n_users + 1):
            if id_user % 1000 == 0:
                print("{}/{}".format(id_user, n_users))
            id_movies = data[:, 1][data[:, 0] == id_user]
            id_ratings = data[:, 2][data[:, 0] == id_user]
            ratings = np.zeros(n_movies)
            ratings[id_movies - 1] = id_ratings
            userItemMatrix.append(list(ratings))
        return userItemMatrix

    def loss(self, groundTruthData, P: NDArray, Q: NDArray) -> float:
        squaredErrors = 0.0
        numOfPrediction = 0

        for u in range(self.n_users):
            for i in range(self.n_items):
                if groundTruthData[u][i] > 0:
                    squaredErrors += pow(
                        groundTruthData[u][i] - self.prediction(P, Q, u, i), 2
                    )
                    numOfPrediction += 1

        return squaredErrors / numOfPrediction

    def optimize(self, error:float, id_user:int, id_item:int):
        # Johnny
        self._P[id_user, :] += self.lr * (2 * error * self._Q[id_item,:] - 2 * self.weight_decay * self._P[id_user,:])
        self._Q[id_item, :] += self.lr * (2 * error * self._P[id_user,:] - 2 * self.weight_decay * self._Q[id_item,:])
        