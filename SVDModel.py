# SVD
## Import libraries
from lib.models import RecommendSystemModel

from typing import List, Any, Tuple, Union
from numpy.typing import NDArray
import pandas as pd
import numpy as np

import time

from numba import njit, prange

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
        userItemMatrix = (
            self.data[["userId", "movieId", "rating"]]
            .pivot_table(columns="movieId", index="userId", values="rating")
            .fillna(0)
        )

        userItemMatrix = userItemMatrix.to_numpy()
        print(f"User Item Matrix Shape: {userItemMatrix.shape}")
        print(f"User Reference length: {self.n_users}")
        print(f"Item Reference length: {self.n_items}")

        trainBeforeSplit = userItemMatrix.copy()
        trainBeforeSplit.fill(0)

        self.train = trainBeforeSplit.copy()
        self.valid = trainBeforeSplit.copy()
        self.test = trainBeforeSplit.copy()

        for i in range(self.n_users):
            for j in range(self.n_items):
                if userItemMatrix[i, j]:
                    if np.random.binomial(1, ratio_train_test, 1):
                        if np.random.binomial(1, ratio_train_valid, 1):
                            self.train[i, j] = userItemMatrix[i, j]
                        else:
                            self.valid[i, j] = userItemMatrix[i, j]
                    else:
                        self.test[i, j] = userItemMatrix[i, j]

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

        # create reference of users and movies
        self.users_ref = self.data["userId"].unique()
        self.users_ref.sort()
        self.movies_ref = self.data["movieId"].unique()
        self.movies_ref.sort()

        self.n_users = len(self.users_ref)
        self.n_items = len(self.movies_ref)

    def train_one_epoch(self):
        return _train_one_epoch(
            self.n_users,
            self.n_items,
            self.train,
            self._P,
            self._Q,
            self.mode,
            self._mean,
            self._bu,
            self._bi,
            self.lr,
            self.weight_decay,
        )

    def training(self) -> Tuple[NDArray, NDArray, float, float]:
        loss_train = []
        loss_valid = []
        errors = []

        self._P = np.random.rand(self.n_users, self.features) * 0.1
        self._Q = np.random.rand(self.n_items, self.features) * 0.1
        self._bu = np.zeros(self.n_users)
        self._bi = np.zeros(self.n_items)
        self._mean = 0
        if self.mode == "svd++":
            # for advanced SVD
            self._bu = np.zeros(self.n_users)
            self._bi = np.zeros(self.n_items)
            self._mean = np.mean(self.data["rating"])
        # Johnny
        tic = time.perf_counter()
        for e in range(self.epochs):
            error = self.train_one_epoch()
            errors.append(error)

            trainLoss = self.loss(self.train)
            validLoss = self.loss(self.valid)
            loss_train.append(trainLoss)
            loss_valid.append(validLoss)

            if e % 10 == 0:
                toc = time.perf_counter()
                print(
                    "Epoch : ",
                    "{:3.0f}".format(e + 1),
                    " | Train :",
                    "{:3.3f}".format(trainLoss),
                    " | Valid :",
                    "{:3.3f}".format(validLoss),
                    " | Time :",
                    "{:3.5f}s".format(toc - tic),
                )
                tic = time.perf_counter()

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
        return _prediction(
            self.id_user,
            self.id_item,
            self._P,
            self._Q,
            self.mode,
            self._mean,
            self._bu,
            self._bi,
        )

    def loss(self, groundTruthData) -> float:
        # Woody
        return _loss(
            groundTruthData,
            self.n_users,
            self.n_items,
            self.mode,
            self._P,
            self._Q,
            self._mean,
            self._bu,
            self._bi,
            self.train,
        )


@njit(parallel=True, fastmath=True)
def _train_one_epoch(
    n_users, n_items, train, _P, _Q, mode, _mean, _bu, _bi, lr, weight_decay
):
    error = 0
    for id_user in prange(n_users):
        for id_item in prange(n_items):
            if train[id_user, id_item] > 0:
                # Predict
                predict = _prediction(id_user, id_item, _P, _Q, mode, _mean, _bu, _bi)

                error = train[id_user, id_item] - predict

                if mode == "svd++":
                    _bu[id_user] += lr * (error - weight_decay * _bu[id_user])
                    _bi[id_item] += lr * (error - weight_decay * _bi[id_item])

                _P[id_user, :] += lr * (
                    error * _Q[id_item, :] - weight_decay * _P[id_user, :]
                )
                _Q[id_item, :] += lr * (
                    error * _P[id_user, :] - weight_decay * _Q[id_item, :]
                )

    return error


# Woody
@njit(parallel=True, fastmath=True)
def _loss(
    groundTruthData,
    n_users,
    n_items,
    mode,
    _P,
    _Q,
    _mean,
    _bu,
    _bi,
    train,
):
    squaredErrors = 0.0
    numOfPrediction = 0
    for u in prange(n_users):
        for i in prange(n_items):
            if groundTruthData[u, i] > 0:
                predict = np.dot(_P[u, :], _Q[i, :])
                if mode == "svd++":
                    predict += _mean + _bu[u] + _bi[i]
                squaredErrors += pow(groundTruthData[u, i] - predict, 2)
                numOfPrediction += 1
    return 0 if numOfPrediction == 0 else squaredErrors / numOfPrediction


@njit(parallel=True, fastmath=True)
def _prediction(u: int, i: int, _P, _Q, mode, _mean, _bu, _bi):
    # Woody
    predict = 0
    for f in prange(len(_P[0])):
        predict += _P[u, f]* _Q[i, f]
    if mode == "svd++":
        predict += _mean + _bu[u] + _bi[i]
    return predict
