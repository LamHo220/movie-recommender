# SVD
## Import libraries
from lib.models import RecommendSystemModel

from typing import List, Any, Tuple, Union
from numpy.typing import NDArray
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from joblib import Parallel, delayed
import time

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

#         for label in range(1, self.n_items):
#             if label not in userItemMatrix.columns:
#                 userItemMatrix[label] = 0
#         userItemMatrix = userItemMatrix[sorted(userItemMatrix.columns)].to_numpy()
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
        
    def _process(self,id_user,id_item):
        predict = self.prediction(id_user, id_item)
        error = self.train[id_user, id_item] - predict
        self.optimize(error, id_user, id_item)
        return error
            
    def _run(self,id_user, id_item):
        return self._process(id_user,id_item)
        
    def _train_one_epoches(self):
        return [self._run(id_user, id_item) for id_user in range(self.n_users) for id_item in range(self.n_items) if self.train[id_user, id_item] > 0]

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
        tic = time.perf_counter()
        for e in range(self.epochs):
            _errors = self._train_one_epoches()
            errors += _errors
            
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
                    " | Time :", "{:3.5f}s".format(toc-tic)
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
        predict = np.dot(self._P[u, :], self._Q[i, :])
        if self.mode == "svd++":
            predict += self._mean + self._bu[u] + self._bi[i]
        return predict

    def loss(self, groundTruthData) -> float:
        # Woody
        squaredErrors = 0.0
        numOfPrediction = 0
        for u in range(self.n_users):
            for i in range(self.n_items):
                if groundTruthData[u, i] > 0:
                    squaredErrors += pow(
                        groundTruthData[u, i] - self.prediction(u, i), 2
                    )
                    numOfPrediction += 1
        return 0 if numOfPrediction==0 else squaredErrors / numOfPrediction

    def optimize(self, error: float, id_user: int, id_item: int):
        # Johnny
        self._P[id_user, :] += self.lr * (
            error * self._Q[id_item, :] - self.weight_decay * self._P[id_user, :]
        )
        self._Q[id_item, :] += self.lr * (
            error * self._P[id_user, :] - self.weight_decay * self._Q[id_item, :]
        )