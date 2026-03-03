import numpy as np
from nnfs.datasets import spiral_data
import nnfs
nnfs.init()

class Dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weight = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros(1, n_neurons)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weight) + self.biases

class Activation_Relu:
    
