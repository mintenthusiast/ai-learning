import numpy as np
import os

class Layer:
    def __init__(self, dims, function):
        self.dims = dims
        self.function = function
        
        self.weight_matrix = self._weight_init(dims)
        self.biases = np.zeros((1, dims[1]))
    
    def _weight_init(self, dims):
        std = np.sqrt(2.0 / dims[0])
        weights = np.random.randn(dims[0], dims[1]) * std
        return weights

class MLP:
    def __init__(self, lr=0.001, hidden_layers=2, h_sizes=[512, 256], input_size=784, output_size=10, load=False): # 2 + 1 output layer
        self.lr = lr
        self.no_layers = hidden_layers + 2
        self.layers = []

        if len(h_sizes) != hidden_layers:
            raise ValueError

        self.layer_sizes = [input_size]

        for x in h_sizes:
            self.layer_sizes.append(x)

        self.layer_sizes.append(output_size)

        self.forward_cache = []
        self.delta_cache = []

        for i in range(1, self.no_layers):
            dims = (self.layer_sizes[i - 1], self.layer_sizes[i])
            newLayer = Layer(dims=dims, function=self.relu)
            self.layers.append(newLayer)
        
        self.layers[-1].function = self.softmax
        
    def forward(self, x):
        self.forward_cache = [(None, x)]
        curr_input = x

        for layer in self.layers:
            z_i = np.dot(curr_input, layer.weight_matrix) + layer.biases
            a_i = layer.function(z_i)

            curr_input = a_i
            self.forward_cache.append((z_i, a_i))

        return self.forward_cache[-1][1]

    def backward(self, x, y):
        self.delta_cache = []
        # x: (batch_size, input_size)

        m = x.shape[0]
        a_out = self.forward_cache[-1][1]

        d_out = (a_out - y) / m # with cross entropy
        prev_d = d_out

        for i in reversed(range(len(self.layers))):
            z_i, a_i = self.forward_cache[i + 1]
            a_prev = self.forward_cache[i][1]
            layer = self.layers[i]

            if i == len(self.layers) - 1:
                dL_dz = prev_d
            else:
                dL_dz = prev_d * self.d_relu(z_i)

            dL_dW = np.dot(a_prev.T, dL_dz)
            dL_db = np.sum(dL_dz, axis=0, keepdims=True)
            prev_d = np.dot(dL_dz, layer.weight_matrix.T)

            delta_w = self.lr * dL_dW
            delta_b = self.lr * dL_db

            self.delta_cache.append((delta_w, delta_b))

            layer.weight_matrix -= delta_w
            layer.biases -= delta_b
    
    def fit(self, x, y, epochs, batch_size, shuffle=True):
        samples = len(x)

        for epoch in range(epochs):
            idx = np.arange(samples)

            if shuffle:
                np.random.shuffle(idx)

            for start in range(0, samples, batch_size):
                batch_idx = idx[start:start + batch_size]
                batch_x = x[batch_idx]
                batch_y = y[batch_idx]
                
                p = self.forward(batch_x)
                self.backward(batch_x, batch_y)

            p = self.forward(x)
            loss = np.mean(self.cross_entropy_loss(p, y))
            acc = np.mean(np.argmax(p, axis=1) == np.argmax(y, axis=1))

            print(f"Epoch: {epoch}, Loss: {loss:.8f}, Accuracy: {acc:.8f}")
        
        print("Done Training")

    def save(self, path):
        saves = {}

        for i, layer in enumerate(self.layers):
            saves[f"w_{i}"] = layer.weight_matrix
            saves[f"b_{i}"] = layer.biases
        
        np.savez(path, **saves)
    
    def load(self, path):
        data = np.load(path)
        for i, layer in enumerate(self.layers):
            layer.weight_matrix = data[f"w_{i}"]
            layer.biases = data[f"b_{i}"]

    def predict(self, x):
        p = self.forward(x)
        return np.argmax(p, axis=1)

    def relu(self, x):
        return np.maximum(0, x).astype(float)

    def d_relu(self, x):
        return np.where(x > 0, 1, 0)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, p, y):
        loss = -np.sum(y * np.log(p + 1e-15), axis=1)
        return loss


mlp = MLP(lr=0.01, hidden_layers=4, h_sizes=[512, 256, 128, 64], input_size=784,
          output_size=10)

script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.normpath(os.path.join(script_dir, '..', 'data', 'mnist.npz'))
data = np.load(path)

x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

y_train_onehot = np.zeros((len(y_train), 10))
y_train_onehot[np.arange(len(y_train)), y_train] = 1

y_test_onehot = np.zeros((len(y_test), 10))
y_test_onehot[np.arange(len(y_test)), y_test] = 1


# mlp.fit(x_train, y_train_onehot, epochs=10, batch_size=32, shuffle=True)

save_path = os.path.normpath(os.path.join(script_dir, '..', 'save', 'save.npz'))
# mlp.save(save_path)

mlp.load(save_path)

single_pred = mlp.predict(x_test[0:1])
print(single_pred)

y_pred = mlp.predict(x_test)
test_acc = np.mean(y_pred == y_test)
print(f"Test Accuracy: {test_acc:.4f}")

