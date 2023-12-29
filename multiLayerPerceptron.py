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

        self.weights_l1 = np.random.rand(params.hidden_size1, params.input_size)
        self.bias_l1 = np.zeros((params.hidden_size1, 1))
        self.weights_l2 = np.random.rand(params.hidden_size2, params.hidden_size1)
        self.bias_l2 = np.zeros((params.hidden_size2, 1))
        self.weights_l3 = np.random.rand(params.output_size, params.hidden_size2)
        self.bias_l3 = np.zeros((params.output_size, 1))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X):
        l1_input = np.dot(self.weights_l1, X) + self.bias_l1
        l1_output = self.sigmoid(l1_input)

        l2_input = np.dot(self.weights_l2, l1_output) + self.bias_l2
        l2_output = self.sigmoid(l2_input)

        output = np.dot(self.weights_l3, l2_output) + self.bias_l3
        return output

    def train(self, X, y, method='gradient', lr=0.01, epochs=1000):
        if method == 'gradient':
            self._gradient(X, y, lr, epochs)
        elif method == 'evolutionary':
            self._evolutionary()
        else:
            raise ValueError("Unsupported optimization method")

    def _gradient(self, X, y, lr, epochs):
        pass

    def _evolutionary(self):
        pass
