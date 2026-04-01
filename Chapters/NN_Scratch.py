import numpy as np
import keras

# class for Layer of neurons
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Defining the starting values for weights and biases randomly but with 0.01 scale
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) 
        self.biases = np.zeros((1, n_neurons))

    # Passing Forward methos
    def forward(self, inputs):
        self.inputs = inputs
        # Applying the dot of (input*weight)+b
        self.output = np.dot(self.inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradient on parameters
        self.dvalues = dvalues
        self.weights = np.dot(self.inputs.T, self.dvalues)
        self.dinputs = np.dot(dvalues, self.weights.T)
        self.dbiases = np.sum(dvalues, keepdims=True, axis=0)

# class for Rectified Linear Activation function
class ReLU:
    # 0 if input is <= 0 else remains same 
    # Forward pass
    def forward(self, z):
        self.inputs = z
        self.output = np.maximum(0, self.inputs)

    # Backward pass
    def backward(self, dvalues):
        self.inputs = dvalues.copy()
        self.dinputs[self.dinputs <= 0 ] = 0

# class for output layer
class Softmax:
    # Forward pass
    def forward(self, inputs):
        # Getting normalized probablities
        exp_values = np.exp(inputs-np.max(input, axis=1, keepdims=True))
        
        # Normalizing them for each samples
        probablities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probablities

    # Backward pass
    def backward(self, dvalues):
        # uninitialized arrays
        self.dinputs = np.zeros_like(dvalues)

        # Enumerate outputs and gradients

        for index, (sinlge_output, sinlge_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1,1)
            # calculation of the Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # Calculate sample-wise gradient
            self.dinputs[index] = np.dot(jacobian_matrix, sinlge_dvalues)

# Loss class
class Loss:
    def calculate(self, output, y ):
        sample_loss = self.forward(output, y)
        # Mean loss
        data_loss = np.mean(sample_loss)
        return data_loss
    
# class of Loss_CategoricalCrossentrophy
class Loss_CategoricalCrossentrophy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        sample = len(y_pred)
        # Clip data to prevent 0
        # clip both sides to not drag mean towards any values
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        #probablities for target valuees
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(sample), y_true]

        # one hot encoded labels only
        if len(y_true.shape) == 2:
            correct_confidences = y_pred_clipped[y_pred_clipped * y_true]

        # Losses
        negative_loss_likelihoods = -np.log(correct_confidences)
        return negative_loss_likelihoods
        
    # Backward pass
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # calculate the gradients
        self.dinputs = -y_true/dvalues
        # normalize gradients
        self.dinputs = self.dinputs/samples

# class mixing bth activation and loss function object
class Activation_Softmax_Loss_CategoricalCrossentrophy():
    # Defining the class
    def __init__(self):
        self.activation = Softmax()
        self.loss = Loss_CategoricalCrossentrophy()

    # Forward pass
    def forward(self, inputs, y_true):
        # output layer activation
        self.activation.forward(inputs)
        self.output = self.activation.output
        # Calculate and return the loss value
        return self.loss.calculate(self.output, y_true)
    
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If labels are one hot encoded, thrun them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so that it can be safely modified
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -=1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Optimizing the accuracy using SGD
class Optimizer_SGD:
    # initializing optimizer
    # learning rate
    def __init__(self, learning_rate=1):
        self.learning_rate = learning_rate

    # updating the parameters weight and biases
    def update_params(self, layer):
        layer.weights += self.learning_rate * layer.dweights
        layer.biases += self.learning_rate * layer.dbiases

