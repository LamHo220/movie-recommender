from abc import abstractmethod, ABC
from typing import Any, Callable, List, Tuple
import numpy as np
from numpy.typing import NDArray
import pandas as pd

""" 
The idea of the construction of abstract class is came from https://colab.research.google.com/drive/1I5CoVTCmwKUMfV6iiGG5Emuk2OtrcgRT?usp=sharing#scrollTo=Ft_CJd64z8Bu&uniqifier=1
"""


class RecommendSystemModel(ABC):
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
        pass
    
    # @abstractmethod
    def split(
        self, ratio_train_test: float, ratio_train_valid: float, tensor: bool = False
    ) -> List[NDArray]:
        pass

    # @abstractmethod
    def data_loader(
        self,
        path: str = None,
        nrows: int = None,
        skiprows=None,
        data: pd.DataFrame = None,
        n_users: int = None,
        n_items=None,
    ) -> None:
        pass

    # @abstractmethod
    def training(self) -> Tuple[NDArray, NDArray, float, float]:
        pass

    # @abstractmethod
    def prediction(self, P: NDArray, Q: NDArray, u: int, i: int) -> float:
        pass

    # @abstractmethod
    def loss(self, groundTruthData, P: NDArray, Q: NDArray) -> float:
        pass

    def optimize(self, error: float, id_user: int, id_item: int, weight_decay):
        pass
