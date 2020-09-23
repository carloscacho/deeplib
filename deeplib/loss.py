""" 
A loss function measures how good our predictions are,
we can use this to adjust the parameters of our netwrk

Loss Function mede quao bom é sua predição,
com isso, você pode ajustar os paramanetros da sua rede

"""
import numpy as np 
from deepLib.tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError

class MSE(Loss):
    """
    MSE é o erro medio quadrado, ou seja,
    ele faz o calculo do erro medio ao quadrado
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)