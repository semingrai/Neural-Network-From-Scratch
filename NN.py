import numpy as np
from nnfs.datasets import spiral_data
import nnfs
nnfs.init()

# Neuron creating class
class Dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weight = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    # Forwarding the input values towards the hidden layers
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weight) + self.biases

# ReLU activation function class(0-x)
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# Softmax function for output measures and probablity
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        probablities =  exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probablities

# Common loss class
class Loss:
    def calculate(self, output, y):
        # calculate sample loss
        sample_loss = self.forward(output, y)
        # calculate mean loss
        data_loss = np.mean(sample_loss)
        return data_loss


# Loss Entrophy class
class Loss_Categorical_Crossentrophy(Loss):
    #Forward pass
    def forward(self, y_pred, y_true):
        # number of sample in a batch
        sample = len(y_pred)
        # Clip data to prevent 0 and error
        # Clip both sides not to drag mean towards any values not being biases
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(sample),
                y_true
            ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        neg_log_likelihoods = -np.log(correct_confidences)
        return neg_log_likelihoods
        

X, y = spiral_data(samples=100, classes=3)

# 2 inputs 1 3 neuron hidden layers
dense1 =  Dense_Layer(2,3)
activation1 = Activation_ReLU()

dense2 = Dense_Layer(3,3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss_function = Loss_Categorical_Crossentrophy()
loss = loss_function.calculate(activation2.output, y)
print("Loss:", loss)
