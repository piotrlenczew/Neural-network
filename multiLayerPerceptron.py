import numpy as np


class MultiLayerPerceptronParams:
    def __init__(
        self,
        input_size: int = 1,
        hidden_sizes: [int] = [64, 32],
        output_size: int = 1
    ):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size


class MultiLayerPerceptron:
    def __init__(self, params: MultiLayerPerceptronParams):
        self.params = params

        np.random.seed(42)
        self.weights = [np.random.randn(params.input_size, params.hidden_sizes[0])]
        self.biases = [np.zeros((1, params.hidden_sizes[0]))]

        for i in range(len(params.hidden_sizes)-1):
            self.weights.append(np.random.randn(params.hidden_sizes[i], params.hidden_sizes[i+1]))
            self.biases.append(np.zeros((1, params.hidden_sizes[i+1])))

        self.weights.append(np.random.randn(params.hidden_sizes[-1], params.output_size))
        self.biases.append(np.zeros((1, params.output_size)))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, X, y, method='gradient', lr=0.00005, epochs=3000, population_size=50, generations=100):
        if method == 'gradient':
            losses = self._gradient(X, y, lr, epochs)
            return losses
        elif method == 'evolutionary':
            self._evolutionary(X, y, population_size, generations)
        else:
            raise ValueError("Unsupported optimization method")

    def _gradient(self, X, y, lr, epochs):
        losses = []
        for epoch in range(epochs):
            # Forward pass
            layer_outputs = [X]
            for i in range(len(self.params.hidden_sizes)):
                layer_outputs.append(self.sigmoid(np.dot(layer_outputs[i], self.weights[i]) + self.biases[i]))
            layer_outputs.append(np.dot(layer_outputs[-1], self.weights[-1]) + self.biases[-1])
            output = layer_outputs[-1]

            # Calculating loss
            loss = np.mean((output - y)**2)/2
            losses.append(loss)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

            # Backward pass
            errors = [output - y]
            for i in range(len(self.params.hidden_sizes), 0, -1):
                errors.insert(0, np.dot(errors[0], self.weights[i].T) * self.sigmoid_derivative(layer_outputs[i]))

            # Updating weights and biases
            for i in range(len(self.params.hidden_sizes)+1):
                self.weights[i] -= lr * np.dot(layer_outputs[i].T, errors[i])
                self.biases[i] -= lr * np.sum(errors[i], axis=0, keepdims=True)
        return losses

    def _evolutionary(self, X, y, population_size=50, generations=100):
        pass

    def predict(self, X):
        layer_outputs = [X]
        for i in range(len(self.params.hidden_sizes)):
            layer_outputs.append(self.sigmoid(np.dot(layer_outputs[i], self.weights[i]) + self.biases[i]))
        layer_outputs.append(np.dot(layer_outputs[-1], self.weights[-1]) + self.biases[-1])
        output = layer_outputs[-1]
        return output