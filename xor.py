"""
the canonical exaple of a function that can't be learned
with a simple linear model is XOR

XOR é um exemplo comum que não pode ser aprendido com um
modelo linear 
"""

import numpy as np
from deeplib.train import train
from deeplib.nn import NeuralNet
from deeplib.layers import Linear, Tanh

inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]) 

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

net = NeuralNet([
    Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2)
])

train(net, inputs, targets)

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)