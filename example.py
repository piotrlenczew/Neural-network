import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Define the target function
def target_function(x):
    return x**2 * np.sin(x) + 100 * np.sin(x) * np.cos(x)

# Generate data for training
np.random.seed(42)
X_train = np.random.uniform(-10, 10, 1000).reshape(-1, 1)
y_train = target_function(X_train)

# Define the architecture of the MLP
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='sigmoid', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='sigmoid'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=2)

# Generate test data
X_test = np.linspace(-10, 10, 1000).reshape(-1, 1)

# Make predictions using the trained model
y_pred = model.predict(X_test)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(X_test, target_function(X_test), label='True Function', linestyle='--', color='blue')
plt.plot(X_test, y_pred, label='MLP Approximation', color='red')
plt.legend()
plt.title('Approximating a Function with MLP')
plt.xlabel('x')
plt.ylabel('y')
plt.show()