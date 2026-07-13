import numpy as np
import matplotlib.pyplot as plt

x_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=20):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.errors_per_epoch = []

    def predict(self, in_data):
        out = np.dot(in_data, self.weights) + self.bias
        return np.where(out >= 0, 1, 0)
    
    def learn(self, in_data, target_data):
        self.weights = np.zeros(in_data.shape[1])
        self.bias = 0.0
        for i in range(self.epochs):
            errors = 0
            for xi, target in zip(in_data, target_data):
                prediction = self.predict(xi)
                update = self.lr * (target - prediction)
                self.weights += update * xi
                self.bias += update
                errors += int(update != 0.0)
            self.errors_per_epoch.append(errors)

p_or = Perceptron(learning_rate=0.1, epochs=100)
p_or.learn(x_or, y_or)

print("weights:", p_or.weights)
print("bias", p_or.bias)
print("Predictions", p_or.predict(x_or))
