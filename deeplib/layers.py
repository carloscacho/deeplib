"""
Our neural nets will be made up of layers.
Each layer needs to pass its input forward
and propagate gradients backward. For Example,
a neural net might look like 

inputs -> linear  -> tanh -> linear -> output

As redes neurais serão feitas de camadas (layers).
Cada camada (layer) precisa passar sua entrada para frente
e propagar gradientes para trás. Por exemplo,
uma rede neural pode parecer

entrada -> linear  -> tanh -> linear -> saida
"""
from typing import Dict, Callable
import numpy as np 
from deeplib.tensor import Tensor

class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Produz as saidas (output) correspondentes as entradas (inputs)
        """
        raise NotImplementedError

    
    def backward(self, grad: Tensor) -> Tensor:
        """
        backpropagate this gradiente through the layer
        retropropagar este gradiente através da camada (layer)
        """
        raise NotImplementedError

class Linear(Layer):
    """
    computa as saidas = entradas @ w + b
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        #inputs será (batch_size, input_size)
        #output será (batch_size, output_size)
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)


    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        # @ é uma multiplicação de matizes
        return inputs @ self.params["w"] + self.params["b"]


    def backward(self, grad: Tensor) -> Tensor:
        """"
        if y = f(x) and x = a * b + c
        então dy/da = f'(x) * b
        e dy/db = f'(x) * a
        e dy/dc = f'(x)

        if y = f(x) and x = a @ b + c
        então dy/da = f'(x) @ b.T
        e dy/db = a.T @ f'(x)
        e dy/dc = f'(x)
        """
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T

F = Callable[[Tensor],Tensor]

class Activation(Layer):
    """
    Activation Layer trata-se de aplicar uma função
    elemento a elemento das entradas(inputs)
    """
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        se y = f(x) e x = g(z)
        então dy/dz = f'(x) * g'(z)
        """
        return self.f_prime(self.inputs) * grad

def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    y = np.tanh(x)
    return 1 - y ** 2


class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)