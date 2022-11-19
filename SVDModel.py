# SVD
## Import libraries
from lib.models import RecommendSystemModel

from typing import List, Any, Tuple, Union
from numpy.typing import NDArray
import pandas as pd
import numpy as np

import time

from numba import njit, prange, typed, types

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
        self, ratio_train_test: float, ratio_train_valid: float
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

        self.train, self.valid, self.test =  _split(ratio_train_test, ratio_train_valid, userItemMatrix, self.n_users, self.n_items)

    def data_loader(
        self,
        path: str = None,
        nrows: int = None,
        skiprows=None,
        data: pd.DataFrame = None,
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

            if e > 2:
                if abs(validLoss - loss_valid[-2]) < self.stopping:
                    break
        print("Training stopped:")
        testLoss = self.loss(self.test)
        print(
            "Epoch : ",
            "{:3.0f}".format(e + 1),
            " | Train Loss :",
            "{:3.3f}".format(trainLoss),
            " | Valid Loss:",
            "{:3.3f}".format(validLoss),
            " | Test Loss:",
            "{:3.3f}".format(testLoss),
        )

        return loss_train, loss_valid, testLoss, errors

    def prediction(self, u: int, i: int) -> float:
        # Woody
        return _prediction(u, i, self._P, self._Q, self._mean, self._bu, self._bi, self.mode)


    def loss(self, groundTruthData) -> float:
        # Woody
        return _loss(groundTruthData, self.n_users, self.n_items, self.mode, self._P,self._Q, self._mean, self._bu, self._bi,self.train)


@njit(parallel=True, fastmath=True)
def _train_one_epoch(n_users, n_items, train, _P, _Q, mode, _mean, _bu, _bi, lr, weight_decay):
    error = 0
    for id_user in prange(n_users):
        for id_item in prange(n_items):
            if train[id_user, id_item] > 0:
                # Predict
                predict = np.dot(_P[id_user, :], _Q[id_item, :])
                if mode == "svd++":
                    predict += _mean + _bu[id_user] + _bi[id_item]

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
def _loss(groundTruthData, n_users, n_items,mode,_P,_Q,_mean,_bu,_bi,train,):
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
def _prediction( u: int, i: int, _P, _Q, _mean, _bu, _bi, mode):
    # Woody
    predict = np.dot(_P[u, :], _Q[i, :])
    if mode == "svd++":
        predict += _mean + _bu[u] + _bi[i]
    return predict

@njit(parallel=True, fastmath=True)
def _split(
    ratio_train_test: float, ratio_train_valid: float, userItemMatrix, n_users, n_items):

    trainBeforeSplit = userItemMatrix.copy()
    trainBeforeSplit.fill(0)

    train = trainBeforeSplit.copy()
    valid = trainBeforeSplit.copy()
    test = trainBeforeSplit.copy()

    for i in prange(n_users):
        for j in prange(n_items):
            if userItemMatrix[i, j]:
                if np.random.binomial(1, ratio_train_test, 1):
                    if np.random.binomial(1, ratio_train_valid, 1):
                        train[i, j] = userItemMatrix[i, j]
                    else:
                        valid[i, j] = userItemMatrix[i, j]
                else:
                    test[i, j] = userItemMatrix[i, j]
    
    return train, valid, test

def movieRatePredictionByUserIdMovieId(userId, movieId, model):
    if movieId not in model.movies_ref:
        return f"The model does not contain this movieID({movieId})"
    if userId not in model.users_ref:
        return f"The model does not contain this userId({userId})"
    
    p_id = np.where(model.users_ref == userId)[0].item()
    q_id = np.where(model.movies_ref == movieId)[0].item()
    return model.prediction(p_id, q_id)

def topKPrediction(userId, k, model):
    if userId not in model.users_ref:
        return f"The model does not contain this userId({userId})"
        
    predictedRate = []
    for movieId in model.movies_ref:
        predictedRate.append((movieId, movieRatePredictionByUserIdMovieId(userId, movieId, model)))

    predictedRate.sort(key=lambda element: element[1], reverse=True)
    return predictedRate[:k]

@njit(fastmath=True)#(parallel=True, fastmath=True)
def makePrediction( n_users, n_items, test, users_ref, movies_ref, mode, _P, _Q, _mean, _bu, _bi):
    userIds = typed.List.empty_list(types.int64)
    movieIds = typed.List.empty_list(types.int64)
    actuals = typed.List.empty_list(types.float64)
    predictions = typed.List.empty_list(types.float64)
    for i in prange(n_users):
        for j in prange(n_items):
            if not test[i,j] == 0:
                userId = users_ref[i]
                movieId = movies_ref[j]
                prediction = np.dot(_P[i, :], _Q[j, :])
                if mode == "svd++":
                    prediction += _mean + _bu[i] + _bi[j]
                # prediction = movieRatePredictionByUserIdMovieId(userId=userId,movieId=movieId,model=svd)
                # prediction = prediction(i,j)
                userIds.append(userId)
                movieIds.append(movieId)
                actuals.append(test[i,j])
                predictions.append(prediction)
    return userIds, movieIds, actuals, predictions

@njit(fastmath=True)#(parallel=True, fastmath=True)
def findThreshold(df, threshold):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    precision = 0
    recall = 0
    for i in prange(len(df)):
        _,_,actual,prediction = df[i]
        if actual >= threshold:
            if prediction >= threshold:
                true_positive+=1
            else:
                false_negative+=1
        else:
            if prediction >= threshold:
                false_positive+=1
            else:
                true_negative+=1
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive!=0 else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative!=0 else 0
    f1 = (2 * precision*recall ) / (precision + recall)
    return precision, recall, f1