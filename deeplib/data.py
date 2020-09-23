"""
we'll feed inputs into out network in batches.
So here are some tools for iterating over data in batches.

iremos alimentar as entradas em nossa rede neurais(NeuralNet) em lotes.
Portanto, aqui estão algumas ferramentas para iterar os dados em lotes.
"""
from typing import Iterator, NamedTuple
import numpy as np

from deeplib.tensor import Tensor 

Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])

class DataIterator:
    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        raise NotImplementedError

class BatchIterator(DataIterator):
    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        starts = np.arange(0, len(inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)
        
        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            yield Batch(batch_inputs, batch_targets)