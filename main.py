import numpy as np
import matplotlib.pyplot as plt
from multiLayerPerceptron import MultiLayerPerceptronParams, MultiLayerPerceptron


def target_function(x):
    return x ** 2 * np.sin(x) + 100 * np.sin(x) * np.cos(x)


np.random.seed(42)
X_train = np.random.uniform(-10, 10, 1000).reshape(-1, 1)
y_train = target_function(X_train)
X_test = np.linspace(-10, 10, 1000).reshape(-1, 1)

mlpParams = MultiLayerPerceptronParams()
mlp = MultiLayerPerceptron(mlpParams)
losses = mlp.train(X_train, y_train, method='gradient')
plt.plot(losses)
plt.title(f"loss per epochs")
plt.show()

y_pred = mlp.predict(X_test)
plt.figure(figsize=(10, 6))
plt.plot(X_test, target_function(X_test), label='True Function', linestyle='--', color='blue')
plt.plot(X_test, y_pred, label='MLP Approximation', color='red')
plt.legend()
plt.title('Approximating a Function with MLP')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
