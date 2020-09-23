"""
FizzBuzz is the following problem:
For each of the numbers 1 to 100:
* if the number is divisible by 3, print "fizz"
* if the number is divisible by 5, print "buzz"
* if the number is divisible by 15, print "fizzbuzz"
* otherwise, just print the number

FizzBuzz é o seguinte problema:
para cada numero de 1 a 100:
* se o numero é divisivel por 3, print "fizz"
* se o numero é divisivel por 5, print "buzz"
* se o numero é divisivel por 15, print "fizzbuzz"
* caso contrario, print o numero

"""
from typing import List
import numpy as np

from deeplib.train import train
from deeplib.nn import NeuralNet
from deeplib.layers import Linear, Tanh
from deeplib.optim import SGD

def fiz_buzz_enconde(x: int) -> List[int]:
    if x % 15 == 0:
        return [0,0,0,1]
    elif x % 5 == 0:
        return [0,0,1,0]
    elif x % 3 == 0:
        return [0,1,0,0]
    else:
        return [1,0,0,0]

def binary_encode(x: int) -> List[int]:
    return [x >> i & 1 for i in range(10)]

inputs = np.array([
    binary_encode(x)
    for x in range(101, 1024)
])

targets = np.array([
    fiz_buzz_enconde(x)
    for x in range(101, 1024)
])

net = NeuralNet([
    Linear(input_size=10, output_size=50),
    Tanh(),
    Linear(input_size=50, output_size=4)
])

train(net, inputs, targets, num_epochs=50000, optimizer=SGD(lr=0.0001))

for x in range(1, 101):
    pred = net.forward(binary_encode(x))
    pred_idx = np.argmax(pred)
    act_idx = np.argmax(fiz_buzz_enconde(x))
    labels = [str(x), "fizz", "buzz", "fizzbuzz"]
    print(x, labels[pred_idx], labels[act_idx])