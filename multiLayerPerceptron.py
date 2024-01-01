import numpy as np
import random


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
        self.weights, self.biases = self._initialize_random_weights_biases()

    def _initialize_random_weights_biases(self):
        weights = [np.random.randn(self.params.input_size, self.params.hidden_sizes[0])]
        biases = [np.zeros((1, self.params.hidden_sizes[0]))]

        for i in range(len(self.params.hidden_sizes)-1):
            weights.append(np.random.randn(self.params.hidden_sizes[i], self.params.hidden_sizes[i+1]))
            biases.append(np.zeros((1, self.params.hidden_sizes[i+1])))

        weights.append(np.random.randn(self.params.hidden_sizes[-1], self.params.output_size))
        biases.append(np.zeros((1, self.params.output_size)))

        return weights, biases

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, X, y, method='gradient', lr=0.00005, epochs=1000, population_size=50, generations=1000):
        if method == 'gradient':
            losses = self._gradient(X, y, lr, epochs)
            return losses
        elif method == 'evolutionary':
            losses = self._evolutionary(X, y, population_size, generations)
            return losses
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

            loss = np.mean((output - y)**2)/2
            losses.append(loss)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

            # Backward pass
            errors = [output - y]
            for i in range(len(self.params.hidden_sizes), 0, -1):
                errors.insert(0, np.dot(errors[0], self.weights[i].T) * self.sigmoid_derivative(layer_outputs[i]))

            for i in range(len(self.params.hidden_sizes)+1):
                self.weights[i] -= lr * np.dot(layer_outputs[i].T, errors[i])
                self.biases[i] -= lr * np.sum(errors[i], axis=0, keepdims=True)
        return losses

    def _evolutionary(self, X, y, population_size, generations):
        losses = []
        population = []
        for _ in range(population_size):
            weights, biases = self._initialize_random_weights_biases()
            population.append((weights, biases))

        for generation in range(generations):
            fitness_scores = []
            for weights, biases in population:
                self.weights = weights
                self.biases = biases
                predictions = self.predict(X)
                fitness = np.mean((predictions - y)**2)/2
                fitness_scores.append(((weights, biases), fitness))

            sorted_population = sorted(fitness_scores, key=lambda x: x[1])
            top_individuals = [individual[0] for individual in sorted_population[:population_size // 2]]

            loss = sorted_population[0][1]
            losses.append(loss)

            offsprings = self._crossover_and_mutate(top_individuals, population_size - len(top_individuals))
            population = top_individuals + offsprings
            if generation % 50 == 0:
                print(f"Generation {generation}, Best Fitness: {sorted_population[0][1]}")
        best_weights, best_biases = sorted_population[0][0]
        self.weights = best_weights
        self.biases = best_biases
        return losses

    def _crossover_and_mutate(self, parents, num_offsprings, mutation_rate=0.01):
        offsprings = []
        for _ in range(num_offsprings):
            parent1, parent2 = random.sample(parents, 2)

            split_point = random.randint(0, len(parent1[0]) - 1)
            offspring_weights = []
            offspring_biases = []

            for i in range(len(parent1[0])):
                if i < split_point:
                    offspring_weights.append(parent1[0][i].copy())
                    offspring_biases.append(parent1[1][i].copy())
                else:
                    offspring_weights.append(parent2[0][i].copy())
                    offspring_biases.append(parent2[1][i].copy())

            for i in range(len(offspring_weights)):
                if random.random() < mutation_rate:
                    offspring_weights[i] += np.random.randn(*offspring_weights[i].shape)
                if random.random() < mutation_rate:
                    offspring_biases[i] += np.random.randn(*offspring_biases[i].shape)

            offsprings.append((offspring_weights, offspring_biases))

        return offsprings

    def predict(self, X):
        layer_outputs = [X]
        for i in range(len(self.params.hidden_sizes)):
            layer_outputs.append(self.sigmoid(np.dot(layer_outputs[i], self.weights[i]) + self.biases[i]))
        layer_outputs.append(np.dot(layer_outputs[-1], self.weights[-1]) + self.biases[-1])
        output = layer_outputs[-1]
        return output
