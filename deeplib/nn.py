"""
A neuralNet is just a collection of layers.
It behaves a lot like a layer itself, although
we're not goif to make it one

Uma rede neural é apenas uma coleção de camadas (layers).
Ela se comporta muito como uma camada(layer), embora
não vamos fazer novamente uma camada
"""
from typing import Sequence

from deeplib.tensor import Tensor
from deeplib.layers import Layer

class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

