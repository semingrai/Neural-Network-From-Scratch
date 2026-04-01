# Neural Network from Scratch — MNIST

A fully manual implementation of a neural network using only NumPy. No PyTorch, no TensorFlow, no autograd. Every forward pass, backward pass, and gradient calculation is written from first principles.

Trained and evaluated on the MNIST handwritten digit dataset. Achieves ~90% test accuracy.

---

## Architecture

```
Input (784) → Dense → ReLU → Dense → Softmax → Cross Entropy Loss
```

- Input layer: 784 neurons (28x28 flattened image)
- Hidden layer: 64 neurons + ReLU activation
- Output layer: 10 neurons (one per digit class) + Softmax
- Loss: Categorical Cross Entropy
- Optimizer: SGD

---

## What is implemented manually

### Dense Layer
Each neuron computes a weighted sum of its inputs plus a bias:

```
output = input * weights + bias
```

Weights are initialized small (`0.01 * randn`) to prevent exploding signals at the start of training. Biases start at zero.

### ReLU Activation
Introduces nonlinearity so the network can learn curved decision boundaries. Without it, stacking Dense layers collapses to a single linear transformation regardless of depth.

```
f(x) = max(0, x)
```

Backward pass passes the gradient through unchanged where input was positive, and blocks it where input was negative or zero.

### Softmax Activation
Converts raw output scores into probabilities that sum to 1 across all classes. Each output depends on all inputs due to the normalization step — this is why its derivative is a matrix (Jacobian) rather than a simple vector.

```
S(x_i) = exp(x_i) / sum(exp(x_j))
```

Inputs are shifted by subtracting the row maximum before exponentiation to prevent numerical overflow.

### Jacobian Matrix
Because every softmax output depends on every input, the derivative cannot be a single number per input. It is a grid of partial derivatives — one for each output-input pair.

For a 3-class output the Jacobian is 3x3:

- Diagonal entries: `S[j] * (1 - S[j])` — self influence
- Off-diagonal entries: `-S[j] * S[k]` — cross influence from normalization

Implemented as:

```python
jacobian = np.diagflat(output) - np.dot(output, output.T)
dinputs = np.dot(jacobian, dvalues)
```

The final dot product applies the chain rule and collapses the 3x3 Jacobian back to a 1D gradient vector for each sample.

### Categorical Cross Entropy Loss
Measures how wrong the predicted probability distribution is relative to the true class.

```
Loss = -log(predicted probability of correct class)
```

High confidence on the correct class produces loss near zero. Low confidence produces high loss. Both sparse integer labels and one-hot encoded labels are supported.

Predicted probabilities are clipped to `[1e-7, 1-1e-7]` to prevent `log(0)` from producing undefined values.

### Combined Softmax + Loss Backward Pass
Running softmax backward then loss backward separately requires the full Jacobian computation. When their derivatives are combined algebraically, almost everything cancels. The result simplifies to:

```
gradient = predicted - true
```

In code:

```python
self.dinputs = dvalues.copy()
self.dinputs[range(samples), y_true] -= 1
self.dinputs = self.dinputs / samples
```

Same mathematical result, no Jacobian loop required.

### SGD Optimizer
Gradient descent — nudge each weight in the direction that reduces loss, scaled by the learning rate.

```
weights -= learning_rate * dweights
biases  -= learning_rate * dbiases
```

Gradients flow backward through the network via the chain rule. Each layer receives the gradient from the layer ahead and passes its own gradient to the layer behind.

---

## Results

- Test accuracy: ~90.8%
- Epochs: converges before 100
- No regularization, no momentum, no learning rate decay

---

## Why build this

Just to have fun. WAU indeed