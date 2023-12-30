import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from multiLayerPerceptron import MultiLayerPerceptronParams, MultiLayerPerceptron


def target_function(x):
    return x ** 2 * np.sin(x) + 100 * np.sin(x) * np.cos(x)


np.random.seed(42)
X = np.random.uniform(-10, 10, 100).reshape(-1, 1)
y = target_function(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlpParams = MultiLayerPerceptronParams()
mlp = MultiLayerPerceptron(mlpParams)
losses = mlp.train(X_train, y_train, method='gradient')
plt.plot(losses)
plt.title(f"loss per epochs")
plt.show()
print(y_test)
print(mlp.predict(X_test))
