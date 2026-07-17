import tensorflow as tf
import numpy as np

class MLP:
    def __init__(self, in_size, out_size, hidden_size, learning_rate=0.001, epochs=100, momentum=0.9, variance=0.999):
        self.weights_in = self.weight_init(in_size, hidden_size)
        self.weights_hidden = self.weight_init(hidden_size, hidden_size)
        self.weights_out = self.weight_init(hidden_size, out_size)
        self.bias_in = np.zeros((1, hidden_size))
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_out = np.zeros((1,out_size))

        self.lr = learning_rate
        self.epochs = epochs

        self.momentum = momentum
        self.variance = variance
        self.t = 0

        self.m_in, self.m_hidden, self.m_out = np.zeros_like(self.weights_in), np.zeros_like(self.weights_hidden), np.zeros_like(self.weights_out)
        self.v_in, self.v_hidden, self.v_out = np.zeros_like(self.weights_in), np.zeros_like(self.weights_hidden), np.zeros_like(self.weights_out)
        self.m_b_in, self.m_b_hidden, self.m_b_out = np.zeros_like(self.bias_in), np.zeros_like(self.bias_hidden), np.zeros_like(self.bias_out)
        self.v_b_in, self.v_b_hidden, self.v_b_out = np.zeros_like(self.bias_in), np.zeros_like(self.bias_hidden), np.zeros_like(self.bias_out)

    def weight_init(self, in_size, out_size):
        # weights = np.random.randn(in_size, out_size) * np.sqrt(2 / (in_size + out_size))
        # weights = np.random.uniform(-1, 1, (in_size, out_size)) * np.sqrt(6 / (in_size + out_size))

        std = np.sqrt(2.0 / in_size)
        weights = np.random.randn(in_size, out_size) * std
        return weights

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def reLu(self, x):
        return np.maximum(0, x).astype(float)
    
    def d_reLu(self, x):
        return np.where(x > 0, 1, 0)
    
    def forward(self, x):
        z_in = np.dot(x, self.weights_in) + self.bias_in
        a_in = self.reLu(z_in)
        z_hidden = np.dot(a_in, self.weights_hidden) + self.bias_hidden
        a_hidden = self.reLu(z_hidden)
        z_out = np.dot(a_hidden, self.weights_out) + self.bias_out
        a_out = self.softmax(z_out)

        return a_out, a_hidden, a_in

    def cross_entropy_loss(self, p, y):
        loss = -np.sum(y * np.log(p + 1e-15), axis=1)
        return loss
    
    def update_momentum(self, momentum, grad):
        m = self.momentum * momentum + (1 - self.momentum) * grad
        return m
    
    def update_variance(self, variance, grad):
        v = (self.variance * variance + (1 - self.variance) * grad ** 2)
        return v
    
    def bias_correct(self, target, constant):
        return target / (1 - constant ** self.t)
    
    def backward(self, x, y):
        m = x.shape[0]
        a_out, a_hidden, a_in = self.forward(x)

        out_delta = (a_out - y) / m

        error_hidden = np.dot(out_delta, self.weights_out.T)
        hidden_delta = error_hidden * self.d_reLu(a_hidden)

        error_in = np.dot(hidden_delta, self.weights_hidden.T)
        in_delta = error_in * self.d_reLu(a_in)

        self.t += 1

        self.m_out = self.update_momentum(self.m_out, np.dot(a_hidden.T, out_delta))
        self.m_hidden = self.update_momentum(self.m_hidden, np.dot(a_in.T, hidden_delta))
        self.m_in = self.update_momentum(self.m_in, np.dot(x.T, in_delta))

        self.v_out = self.update_variance(self.v_out, np.dot(a_hidden.T, out_delta))
        self.v_hidden = self.update_variance(self.v_hidden, np.dot(a_in.T, hidden_delta))
        self.v_in = self.update_variance(self.v_in, np.dot(x.T, in_delta))

        self.m_b_out = self.update_momentum(self.m_b_out, np.sum(out_delta, axis=0, keepdims=True))
        self.m_b_hidden = self.update_momentum(self.m_b_hidden, np.sum(hidden_delta, axis=0, keepdims=True))
        self.m_b_in = self.update_momentum(self.m_b_in, np.sum(in_delta, axis=0, keepdims=True))

        self.v_b_out = self.update_variance(self.v_b_out, np.sum(out_delta, axis=0, keepdims=True))
        self.v_b_hidden = self.update_variance(self.v_b_hidden, np.sum(hidden_delta, axis=0, keepdims=True))
        self.v_b_in = self.update_variance(self.v_b_in, np.sum(in_delta, axis=0, keepdims=True))

        m_hat_out = self.bias_correct(self.m_out, self.momentum)
        m_hat_hidden = self.bias_correct(self.m_hidden, self.momentum)
        m_hat_in = self.bias_correct(self.m_in, self.momentum)
        m_hat_b_out = self.bias_correct(self.m_b_out, self.momentum)
        m_hat_b_hidden = self.bias_correct(self.m_b_hidden, self.momentum)
        m_hat_b_in = self.bias_correct(self.m_b_in, self.momentum)

        v_hat_out = self.bias_correct(self.v_out, self.variance)
        v_hat_hidden = self.bias_correct(self.v_hidden, self.variance)
        v_hat_in = self.bias_correct(self.v_in, self.variance)
        v_hat_b_out = self.bias_correct(self.v_b_out, self.variance)
        v_hat_b_hidden = self.bias_correct(self.v_b_hidden, self.variance)
        v_hat_b_in = self.bias_correct(self.v_b_in, self.variance)

        self.weights_out -= self.lr * m_hat_out / (np.sqrt(v_hat_out) + 1e-8)
        self.weights_hidden -= self.lr * m_hat_hidden / (np.sqrt(v_hat_hidden) + 1e-8)
        self.weights_in -= self.lr * m_hat_in / (np.sqrt(v_hat_in) + 1e-8)

        self.bias_out -= self.lr * m_hat_b_out / (np.sqrt(v_hat_b_out) + 1e-8)
        self.bias_hidden -= self.lr * m_hat_b_hidden / (np.sqrt(v_hat_b_hidden) + 1e-8)
        self.bias_in -= self.lr * m_hat_b_in / (np.sqrt(v_hat_b_in) + 1e-8)

        return a_out

    def fit(self, x, y):
        for epoch in range(self.epochs):
            p = self.backward(x, y)
            if epoch % 10 == 0:
                self.lr *= 0.95
                loss = self.cross_entropy_loss(p, y)
                print(f'Epoch {epoch}, Loss: {np.mean(loss)}, Accuracy: {np.mean(np.argmax(p, axis=1) == np.argmax(y, axis=1))}')

    def predict(self, x):
        p, _, _ = self.forward(x)
        return np.argmax(p, axis=1)



data = np.load('DNN/data/mnist.npz', allow_pickle=True)
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

mlp = MLP(in_size=784, out_size=10, hidden_size=256, learning_rate=0.01, epochs=200, momentum=0.9)
mlp.fit(x_train, tf.keras.utils.to_categorical(y_train, num_classes=10))

y_pred = mlp.predict(x_test)
test_acc = np.mean(y_pred == y_test)
print(f"\nTest Set Classification Accuracy: {test_acc:.4f}")



