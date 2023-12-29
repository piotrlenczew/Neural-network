import numpy as np


class MultiLayerPerceptronParams:
    def __init__(
        self,
        input_size: int = 1,
        hidden_size1: int = 10,
        hidden_size2: int = 5,
        output_size: int = 1
    ):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size


class MultiLayerPerceptron:
    def __init__(self, params: MultiLayerPerceptronParams):
        self.params = params

        np.random.seed(42)
        self.weights_l1 = np.random.randn(params.input_size, params.hidden_size1)
        self.bias_l1 = np.zeros((1, params.hidden_size1))
        self.weights_l2 = np.random.randn(params.hidden_size1, params.hidden_size2)
        self.bias_l2 = np.zeros((1, params.hidden_size2))
        self.weights_l3 = np.random.randn(params.hidden_size2, params.output_size)
        self.bias_l3 = np.zeros((1, params.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def train(self, X, y, method='gradient', lr=0.003, epochs=1000):
        if method == 'gradient':
            losses = self._gradient(X, y, lr, epochs)
            return losses
        elif method == 'evolutionary':
            self._evolutionary()
        else:
            raise ValueError("Unsupported optimization method")

    def _gradient(self, X, y, lr, epochs):
        losses = []
        for epoch in range(epochs):
            # Forward pass
            l1_output = self.sigmoid(np.dot(X, self.weights_l1) + self.bias_l1)
            l2_output = self.sigmoid(np.dot(l1_output, self.weights_l2) + self.bias_l2)
            output = np.dot(l2_output, self.weights_l3) + self.bias_l3

            # Calculating loss
            loss = np.mean((output - y.reshape(-1, 1))**2)/2
            losses.append(loss)

            # Backward pass
            output_error = output - y.reshape(-1, 1)
            l2_error = np.dot(output_error, self.weights_l3.T) * self.sigmoid_derivative(l2_output)
            l1_error = np.dot(l2_error, self.weights_l2.T) * self.sigmoid_derivative(l1_output)

            # Updating weights and biases
            self.weights_l3 -= lr * np.dot(l2_output.T, output_error)
            self.bias_l3 -= lr * np.sum(output_error, axis=0, keepdims=True)
            self.weights_l2 -= lr * np.dot(l1_output.T, l2_error)
            self.bias_l2 -= lr * np.sum(l2_error, axis=0, keepdims=True)
            self.weights_l1 -= lr * np.dot(X.T, l1_error)
            self.bias_l1 -= lr * np.sum(l1_error, axis=0, keepdims=True)
        return losses

    def _evolutionary(self):
        pass

    def predict(self, X):
        l1_output = self.sigmoid(np.dot(X, self.weights_l1) + self.bias_l1)
        l2_output = self.sigmoid(np.dot(l1_output, self.weights_l2) + self.bias_l2)
        output = np.dot(l2_output, self.weights_l3) + self.bias_l3
        return output
