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
    def split(self, ratio: float, tensor: bool = False):
        pass

    # @abstractmethod
    def data_loader(self) -> None:
        pass

    # @abstractmethod
    def learn_to_recommend(self, 
        
        features: int = 10,
        lr: float = 0.0002,
        epochs: int = 101,
        weight_decay: float = 0.02,
        stopping: float = 0.001,
    ) -> Tuple[NDArray, NDArray, float, float]:
        pass

    # @abstractmethod
    def prediction(self, u: int, i: int) -> float:
        pass

    # @abstractmethod
    def loss(self, P: NDArray, Q: NDArray) -> float:
        pass

    