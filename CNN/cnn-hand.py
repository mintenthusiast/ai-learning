import numpy as np
import pickle
import os
import random


class CNN:
    def __init__(self, stride, learning_rate, epochs, hidden_size,
                 kernel_size=3, pool_size=2, pool_stride=2, batch_size=128,
                 momentum=0.9, variance=0.999, kernels=32):
        self.stride = stride
        self.lr = learning_rate
        self.epochs = epochs
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.batch_size = batch_size
        self.momentum = momentum
        self.variance = variance
        self.kernels = kernels
        self.kernel_size = kernel_size

        self.kernel_x, self.kernel_y, self.kernel_z = kernel_size, kernel_size, 3

        self.kernel = np.array([
            self.weight_init(kernel_size * kernel_size, 3).reshape(kernel_size, kernel_size, 3)
            for _ in range(kernels)
        ])
        self.b_kernel = np.zeros(kernels)

        self.m_kernel = np.zeros_like(self.kernel)
        self.v_kernel = np.zeros_like(self.kernel)
        self.m_b_kernel = np.zeros_like(self.b_kernel)
        self.v_b_kernel = np.zeros_like(self.b_kernel)

        self.feature_dim = 7 * 7 * kernels
        self.w_h1 = self.weight_init(self.feature_dim, hidden_size)
        self.w_out = self.weight_init(hidden_size, 10)

        self.b_h1 = np.zeros((1, hidden_size))
        self.b_out = np.zeros((1, 10))

        self.m_h1 = np.zeros_like(self.w_h1)
        self.m_out = np.zeros_like(self.w_out)
        self.v_h1 = np.zeros_like(self.w_h1)
        self.v_out = np.zeros_like(self.w_out)

        self.m_b_h1 = np.zeros_like(self.b_h1)
        self.m_b_out = np.zeros_like(self.b_out)
        self.v_b_h1 = np.zeros_like(self.b_h1)
        self.v_b_out = np.zeros_like(self.b_out)

        self.a_h1, self.a_out, self.raw_a_conv = 0, 0, 0
        self.t = 0

    def convolute(self, image):
        img_x = image.shape[0]
        img_y = image.shape[1]

        f_map_x = int(((img_x - self.kernel_x) / self.stride) + 1)
        f_map_y = int(((img_y - self.kernel_y) / self.stride) + 1)

        f_map = np.zeros((f_map_x, f_map_y, self.kernels))

        for k in range(self.kernels):
            for x in range(f_map_x):
                for y in range(f_map_y):
                    x_start = x * self.stride
                    y_start = y * self.stride

                    section = image[x_start:x_start + self.kernel_x,
                                    y_start:y_start + self.kernel_y]
                    f_map[x, y, k] = np.sum(self.kernel[k] * section) + self.b_kernel[k]

        return f_map

    def d_convolute(self, image, d_conv):
        f_map_x = d_conv.shape[0]
        f_map_y = d_conv.shape[1]

        d_kernel = np.zeros_like(self.kernel)

        for k in range(self.kernels):
            for x in range(f_map_x):
                for y in range(f_map_y):
                    x_start = x * self.stride
                    y_start = y * self.stride

                    section = image[x_start:x_start + self.kernel_size,
                                    y_start:y_start + self.kernel_size]
                    d_kernel[k] += d_conv[x, y, k] * section

        return d_kernel

    def pool(self, f_map):
        f_map_x = f_map.shape[0]
        f_map_y = f_map.shape[1]
        n_kernels = f_map.shape[2]

        out_x = (f_map_x - self.pool_size) // self.pool_stride + 1
        out_y = (f_map_y - self.pool_size) // self.pool_stride + 1

        pooled = np.zeros((out_x, out_y, n_kernels))

        for k in range(n_kernels):
            for x in range(out_x):
                for y in range(out_y):
                    x_start = x * self.pool_stride
                    y_start = y * self.pool_stride

                    section = f_map[x_start:min(x_start + self.pool_size, f_map_x),
                                    y_start:min(y_start + self.pool_size, f_map_y),
                                    k]
                    pooled[x, y, k] = np.max(section)

        return pooled

    def d_pool(self, f_map, d_pooled):
        batch = f_map.shape[0]
        f_map_x = f_map.shape[1]
        f_map_y = f_map.shape[2]
        n_kernels = f_map.shape[3]

        out_x = (f_map_x - self.pool_size) // self.pool_stride + 1
        out_y = (f_map_y - self.pool_size) // self.pool_stride + 1
        d_f_map = np.zeros_like(f_map)

        for b in range(batch):
            for k in range(n_kernels):
                for x in range(out_x):
                    for y in range(out_y):
                        x_start = x * self.pool_stride
                        y_start = y * self.pool_stride
                        section = f_map[b,
                                        x_start:x_start + self.pool_size,
                                        y_start:y_start + self.pool_size,
                                        k]
                        mask = (section == np.max(section))
                        d_f_map[b,
                                x_start:x_start + self.pool_size,
                                y_start:y_start + self.pool_size,
                                k] = d_pooled[b, x, y, k] * mask

        return d_f_map

    def relu(self, x):
        return np.maximum(0, x).astype(float)

    def d_relu(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def update_momentum(self, momentum, grad):
        m = self.momentum * momentum + (1 - self.momentum) * grad
        return m

    def update_variance(self, variance, grad):
        v = (self.variance * variance + (1 - self.variance) * grad ** 2)
        return v

    def bias_correct(self, target, constant):
        return target / (1 - constant ** self.t)

    def cross_entropy_loss(self, p, y):
        loss = -np.sum(y * np.log(p + 1e-15), axis=1)
        return loss

    def weight_init(self, in_size, out_size):
        std = np.sqrt(2.0 / in_size)
        weights = np.random.randn(in_size, out_size) * std
        return weights

    def forward(self, x):
        z_h1 = np.dot(x, self.w_h1) + self.b_h1
        self.a_h1 = self.relu(z_h1)
        z_out = np.dot(self.a_h1, self.w_out) + self.b_out
        self.a_out = self.softmax(z_out)

    def backward(self, x, img, y):
        m = x.shape[0]

        d_out = (self.a_out - y) / m

        err_h1 = np.dot(d_out, self.w_out.T)
        d_h1 = err_h1 * self.d_relu(self.a_h1)

        d_flat = np.dot(d_h1, self.w_h1.T)
        d_pooled = d_flat.reshape(m, 7, 7, self.kernels)
        d_pre_pooled = self.d_pool(self.raw_a_conv, d_pooled)
        d_conv = d_pre_pooled * self.d_relu(self.raw_a_conv)
        d_kernel = np.zeros_like(self.kernel)

        for i in range(m):
            target = img[i].reshape(32, 32, 3)
            d_kernel += self.d_convolute(target, d_conv[i])

        grad_out = np.dot(self.a_h1.T, d_out)
        grad_h1 = np.dot(x.T, d_h1)
        grad_b_out = np.sum(d_out, axis=0, keepdims=True)
        grad_b_h1 = np.sum(d_h1, axis=0, keepdims=True)
        grad_kernel = d_kernel
        grad_b_kernel = np.sum(d_conv, axis=(0, 1, 2))

        self.t += 1

        self.m_out = self.update_momentum(self.m_out, grad_out)
        self.m_h1 = self.update_momentum(self.m_h1, grad_h1)
        self.m_b_out = self.update_momentum(self.m_b_out, grad_b_out)
        self.m_b_h1 = self.update_momentum(self.m_b_h1, grad_b_h1)

        self.m_kernel = self.update_momentum(self.m_kernel, grad_kernel)
        self.m_b_kernel = self.update_momentum(self.m_b_kernel, grad_b_kernel)

        self.v_out = self.update_variance(self.v_out, grad_out)
        self.v_h1 = self.update_variance(self.v_h1, grad_h1)
        self.v_b_out = self.update_variance(self.v_b_out, grad_b_out)
        self.v_b_h1 = self.update_variance(self.v_b_h1, grad_b_h1)

        self.v_kernel = self.update_variance(self.v_kernel, grad_kernel)
        self.v_b_kernel = self.update_variance(self.v_b_kernel, grad_b_kernel)

        m_hat_out = self.bias_correct(self.m_out, self.momentum)
        m_hat_h1 = self.bias_correct(self.m_h1, self.momentum)
        m_hat_b_out = self.bias_correct(self.m_b_out, self.momentum)
        m_hat_b_h1 = self.bias_correct(self.m_b_h1, self.momentum)

        m_hat_kernel = self.bias_correct(self.m_kernel, self.momentum)
        m_hat_b_kernel = self.bias_correct(self.m_b_kernel, self.momentum)

        v_hat_out = self.bias_correct(self.v_out, self.variance)
        v_hat_h1 = self.bias_correct(self.v_h1, self.variance)
        v_hat_b_out = self.bias_correct(self.v_b_out, self.variance)
        v_hat_b_h1 = self.bias_correct(self.v_b_h1, self.variance)

        v_hat_kernel = self.bias_correct(self.v_kernel, self.variance)
        v_hat_b_kernel = self.bias_correct(self.v_b_kernel, self.variance)

        self.w_out -= self.lr * m_hat_out / (np.sqrt(v_hat_out) + 1e-8)
        self.w_h1 -= self.lr * m_hat_h1 / (np.sqrt(v_hat_h1) + 1e-8)

        self.b_out -= self.lr * m_hat_b_out / (np.sqrt(v_hat_b_out) + 1e-8)
        self.b_h1 -= self.lr * m_hat_b_h1 / (np.sqrt(v_hat_b_h1) + 1e-8)

        self.kernel -= self.lr * m_hat_kernel / (np.sqrt(v_hat_kernel) + 1e-8)
        self.b_kernel -= self.lr * m_hat_b_kernel / (np.sqrt(v_hat_b_kernel) + 1e-8)

    def extract(self, x):
        m = x.shape[0]
        features = []
        raw_convs = []

        for i in range(m):
            img = x[i].reshape(32, 32, 3)
            f_map = self.convolute(img)
            raw_convs.append(f_map)
            f_map = self.relu(f_map)
            pooled = self.pool(f_map)
            features.append(pooled.flatten())

        self.raw_a_conv = np.array(raw_convs)
        return np.array(features)

    def fit(self, x, y):
        n_samples = x.shape[0]
        n_batches = n_samples // self.batch_size // 2
        idx = ['', '.', '..', '...']

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            for i in range(n_batches):
                print(f'\rWorking{idx[i % len(idx)]} epoch: {epoch} remaining: {n_batches - i} batches')
                batch_idx = indices[i * self.batch_size:(i + 1) * self.batch_size]
                x_batch = x[batch_idx]
                y_batch = y[batch_idx]
                f_map = self.extract(x_batch)
                self.forward(f_map)
                self.backward(f_map, x_batch, y_batch)

            if epoch % 1 == 0:
                a = random.randint(0, n_samples - self.batch_size)
                x_batch = x[a:a + self.batch_size]
                y_batch = y[a:a + self.batch_size]
                f_map = self.extract(x_batch)
                self.forward(f_map)
                loss = self.cross_entropy_loss(self.a_out, y_batch)
                acc = np.mean(np.argmax(self.a_out, axis=1) == np.argmax(y_batch, axis=1))
                print(f"Epoch: {epoch}, Loss: {np.mean(loss):.4f}, Accuracy: {acc:.4f}")

            if epoch > 0 and epoch % 30 == 0:
                self.lr *= 0.5

    def predict(self, x):
        f_map = self.extract(x)
        self.forward(f_map)
        return self.a_out


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'data', 'cifar-10-batches-py')


def load_batch(filename):
    with open(os.path.join(DATA_DIR, filename), 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    x = np.array(batch[b'data']).reshape(-1, 32 * 32 * 3) / 255.0
    y = np.array(batch[b'labels'])
    return x, y

def to_categorical(y, num_classes=10):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

x_train, y_train = [], []
for i in range(1, 6):
    x, y = load_batch(f'data_batch_{i}')
    x_train.append(x)
    y_train.append(y)

x_train = np.concatenate(x_train, axis=0)
y_train = np.concatenate(y_train, axis=0)

x_test, y_test = load_batch('test_batch')

cnn = CNN(stride=2, learning_rate=0.001, epochs=50,
          hidden_size=256, batch_size=256, kernels=4)

cnn.fit(x_train, to_categorical(y_train, num_classes=10))

y_pred = np.argmax(cnn.predict(x_test), axis=1)
test_acc = np.mean(y_pred == y_test)
print(f"Test Accuracy: {test_acc:.4f}")
