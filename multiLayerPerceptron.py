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

    def train(self, X, y, method='gradient', lr=0.0001, epochs=1000, population_size=50, generations=100):
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

    def _evolutionary(self, X, y, population_size=50, generations=100):
        for generation in range(generations):
            # Generate a population of individuals with random weights and biases
            population = [self._generate_individual() for _ in range(population_size)]

            # Evaluate the fitness of each individual in the population
            fitness_scores = [self._evaluate_fitness(X, y, individual) for individual in population]

            # Select the top-performing individuals to be parents
            parents_indices = np.argsort(fitness_scores)[:int(0.2 * population_size)]
            parents = [population[i] for i in parents_indices]

            # Create a new generation through crossover and mutation
            offspring = self._crossover(parents)
            offspring = self._mutate(offspring)

            # Replace the old population with the new generation
            population[:len(offspring)] = offspring

            # Select the best individual from the current generation as the final model
            best_individual = population[np.argmin(fitness_scores)]

            # Update the neural network with the best individual's weights and biases
            self._update_parameters(best_individual)

    def _generate_individual(self):
        # Generate random weights and biases for the neural network
        individual = {
            'weights_l1': np.random.randn(self.params.input_size, self.params.hidden_size1),
            'bias_l1': np.zeros((1, self.params.hidden_size1)),
            'weights_l2': np.random.randn(self.params.hidden_size1, self.params.hidden_size2),
            'bias_l2': np.zeros((1, self.params.hidden_size2)),
            'weights_l3': np.random.randn(self.params.hidden_size2, self.params.output_size),
            'bias_l3': np.zeros((1, self.params.output_size)),
        }
        return individual

    def _evaluate_fitness(self, X, y, individual):
        # Use mean squared error as the fitness function
        predictions = self.predict(X)
        fitness = np.mean((predictions - y.reshape(-1, 1)) ** 2) / 2
        return fitness

    def _crossover(self, parents):
        # Perform crossover (genetic recombination) to create offspring
        offspring = []
        for _ in range(len(parents) * 2):
            parent1, parent2 = np.random.choice(parents, size=2, replace=False)
            crossover_point = np.random.randint(1, len(parent1) - 1)
            child = {
                k: np.vstack((parent1[k][:crossover_point], parent2[k][crossover_point:]))
                for k in parent1.keys()
            }
            offspring.append(child)
        return offspring

    def _mutate(self, population, mutation_rate=0.1):
        # Perform mutation on the population
        for individual in population:
            for key in individual.keys():
                if np.random.rand() < mutation_rate:
                    individual[key] += np.random.randn(*individual[key].shape) * 0.1
        return population

    def _update_parameters(self, individual):
        # Update the neural network parameters with the best individual's weights and biases
        self.weights_l1 = individual['weights_l1']
        self.bias_l1 = individual['bias_l1']
        self.weights_l2 = individual['weights_l2']
        self.bias_l2 = individual['bias_l2']
        self.weights_l3 = individual['weights_l3']
        self.bias_l3 = individual['bias_l3']

    def predict(self, X):
        l1_output = self.sigmoid(np.dot(X, self.weights_l1) + self.bias_l1)
        l2_output = self.sigmoid(np.dot(l1_output, self.weights_l2) + self.bias_l2)
        output = np.dot(l2_output, self.weights_l3) + self.bias_l3
        return output
