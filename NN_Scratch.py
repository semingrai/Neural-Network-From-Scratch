import numpy as np
#Just using sklearn for the dataset
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')
X = mnist.data.values.astype('float32')
# huge number will make the training explode for weights and biases 
# 255 specificly since it is the actual max pixel value 8 bit
# Dividing itgives exact 0-1 range
X = X/255.0

y = mnist.target.values.astype(int)

class Layers:
    def __init__(self, n_inputs, n_neurons):
        self.weight = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weight) + self.biases

class ReLU:
    def forward(self, input):
        self.output = np.maximum(0, input)

class SoftMax:
    def forward(self, input):
        exp_values = np.exp(input-np.max(input, axis=1, keepdims=True))
        probablities = exp_values/ np.sum(exp_values, axis=1, keepdims=True)
        self.output = probablities

class Loss:
    def calculate(self, output, y):
        sample_loss = self.forward(output, y)
        data_loss = np.mean(sample_loss)
        return data_loss
    
class CrossEntrophyLoss(Loss):
    def forward(self, y_pred, y_true):
        sample = len(y_pred)
        y_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_clipped[
                range(sample),
                y_true
            ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_clipped * y_true, axis=1)
        
        neg_log_likelihoods = -np.log(correct_confidences)
        return neg_log_likelihoods
    
l1 = Layers(784,128)
l1.forward(X)

relu = ReLU()
relu.forward(l1.output)

l2 = Layers(128,10)
l2.forward(relu.output)

softmax = SoftMax()
softmax.forward(l2.output)

entrophy = CrossEntrophyLoss()
loss = entrophy.forward(softmax.output, y)
print(entrophy.calculate(softmax.output, y))

predictions = np.argmax(softmax.output, axis=1)
accuracy = np.mean(predictions == y)
print('acc:', accuracy)