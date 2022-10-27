from abc import abstractmethod, ABC
from typing import Any, Callable, List, Tuple
import numpy as np
from numpy.typing import NDArray

""" 
The idea of the construction of abstract class is came from https://colab.research.google.com/drive/1I5CoVTCmwKUMfV6iiGG5Emuk2OtrcgRT?usp=sharing#scrollTo=Ft_CJd64z8Bu&uniqifier=1
"""


class RecommendSystemModel(ABC):
    def __init__(self) -> None:
        self.data = None

        self.valid = None
        self.train = None

        self.n_users = None
        self.n_items = None

        pass

    # @abstractmethod
    def split(self, ratio: float, tensor: bool = False) -> List[NDArray]:
        """_summary_

        Args:
            ratio (float): How many training data and valid data are loaded
            tensor (bool, optional): It is a tensor or not. Defaults to False.

        Returns:
            List[NDArray]: [training, valid] Training data and valid data in a list
        """
        pass

    # @abstractmethod
    def data_loader(self) -> None:
        pass

    # @abstractmethod
    def learn_to_recommend(self, 
        data: Any,
        features: int = 10,
        lr: float = 0.0002,
        epochs: int = 101,
        weight_decay: float = 0.02,
        stopping: float = 0.001,
    ) -> Tuple[NDArray, NDArray, float, float]:
        """ A method for model to learn to recommend

        Args:
            data (Any): Every evaluation
            features (int, optional): Number of latent variables. Defaults to 10.
            lr (float, optional): Rate for gradient descent. Defaults to 0.0002.
            epochs (int, optional): Number of iterations or maximum loops to perform. Defaults to 101.
            weight_decay (float, optional): L2 regularization to predict rattings different of 0. Defaults to 0.02.
            stopping (float, optional): Scalar associated with the stopping criterion. Defaults to 0.001.

        Returns:
            Tuple[NDArray, NDArray, float, float]: (P, Q, loss_train, loss_valid)
                P: latent matrix of users
                Q: latent matrix of items
                loss_train: vector of the different values of the loss function after each iteration on the train
                loss_valid: vector of the different values of the loss function after each iteration not on valid
        """
        pass

    # @abstractmethod
    def prediction(self, P: NDArray, Q: NDArray, u: int, i: int) -> float:
        """ Calculate the prediction

        Args:
            P (NDArray): user matrix
            Q (NDArray): matrix of items
            u (int): index associated with user u
            i (int): index associated with item i

        Returns:
            float: the predicted evaluation of the user u for the item i
        """
        pass

    # @abstractmethod
    def loss(self, data: Any, P: NDArray, Q: NDArray) -> float:
        """ Calculate the loss

        Args:
            data (Any): ratings
            P (NDArray): matrix of users
            Q (NDArray): matrix of items

        Returns:
            float: mean of squared errors
        """
        pass

    # @abstractmethod
    def svd(self):
        pass
    
    