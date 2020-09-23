"""
Here's a function that can train a neural net
Aqui terá funções para treinar a rede Neural
"""

from deeplib.tensor import Tensor
from deeplib.nn import NeuralNet
from deeplib.loss import Loss, MSE
from deeplib.optim import Optimizer, SGD
from deeplib.data import DataIterator, BatchIterator

def train(net: NeuralNet, 
        inputs: Tensor, 
        targets: Tensor,
        num_epochs: int = 5000,
        itarator: DataIterator = BatchIterator(),
        loss: Loss = MSE(),
        optimizer: Optimizer = SGD()) -> None:
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in itarator(inputs, targets):
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        print(epoch, epoch_loss)