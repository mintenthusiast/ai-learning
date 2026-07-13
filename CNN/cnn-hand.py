import numpy as np
import torchvision

class CNN:
    def __init__(self, stride, learning_rate, epochs, hidden_size):
        self.stride = stride
        self.lr = learning_rate
        self.epochs = epochs

        self.kernel = np.array([[1, 1, 1], 
                               [1, -8, 1],
                               [1, 1, 1]])
        
        self.b_kernel = np.zeros_like(self.kernel)
        
        self.kernel_x, self.kernel_y = 3, 3


        self.w_h1 = self.weight_init(1024, hidden_size)
        self.w_out = self.weight_init(hidden_size, 10)

        self.b_h1 = np.zeros_like(self.w_h1)
        self.b_out = np.zeros_like(self.w_out)

        self.a_h1, self.a_out, self.a_conv = 0, 0, 0
        
        # need to randomise kernel weights maybe

    def convolute(self, image, padding=0):
        # its a square so maybe not but ill keep this for now
        img_x = image.shape()[0]
        img_y = image.shape()[1]
        
        f_map_x = int(((img_x - self.kernel_x + 2 * padding) / self.stride) + 1)
        f_map_y = int(((img_y - self.kernel_y + 2 * padding) / self.stride) + 1)

        f_map = np.zeros((f_map_x, f_map_y))

        if padding != 0:
            image_padded = np.zeros((img_x + padding * 2, img_y + padding * 2))
            image_padded[padding:-1 * padding, padding:-1 * padding] = image
        else:
            image_padded = image

        for y in range(img_y):
            if y > img_y - self.kernel_y:
                break
            elif y % self.stride == 0:
                for x in range(img_x):
                    if x > img_x - self.kernel_x:
                        break
                    elif x % self.stride == 0:
                        section = image_padded[x:x + self.kernel_x, y:y+self.kernel_y]
                        f_map[x, y] = np.sum(self.kernel * section)

        self.a_conv = f_map
        return f_map

    def pool(self, f_map):
        m = f_map.shape()[0] / 4
        pooled = np.zeros((m, m))

        for y in range(m):
            if y > m - 2:
                break
            for x in range(m):
                if x > m - 2:
                    break

                section = f_map[x:x + 2, y:y + 2]
                pooled[x, y] = np.max(section)

        return pooled

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
    

    def weight_init(self, in_size, out_size):
        std = np.sqrt(2.0 / in_size)
        weights = np.random.randn(in_size, out_size) * std
        return weights

    def forward(self, x):
        z_h1 = np.dot(x, self.w_h1) + self.b_h1
        self.a_h1 = self.relu(z_h1)
        z_out = np.dot(self.a_h1, self.w_out) + self.b_out
        self.a_out = self.softmax(z_out)

    def backward(self, x, y):
        m = x.shape[0]
        self.forward(x)

        d_out = (self.a_out - y) / m

        err_h1 = np.dot(d_out, self.w_out.T)
        d_h1 = err_h1 * self.d_relu(self.a_h1)

        err_conv = np.dot(d_h1, self.kernel.T)
        d_conv = err_conv * self.d_relu(self.a_conv)
        
        self.w_out += np.dot(self.a_h1.T, self.d_out) * self.lr
        self.w_h1 += np.dot(x.T, d_h1) * self.lr
        self.kernel += np.dot(self.a_out.T, self.d_conv) * self.lr

        self.b_out += np.sum(d_out, axis=0, keepdims=True) * self.lr
        self.b_h1 += np.sum(d_h1, axis=0, keepdims=True) * self.lr
        self.b_kernel += np.sum(d_conv, axis=0, keepdims=True) * self.lr

    def fit(self, x, y):
        for epoch in range(self.epochs):
            f_map = self.pool(self.relu(self.convolute(x)))
            self.backward(f_map, y)
            if epoch % 10 == 0:
                loss = self.cross_entropy_loss(self.a_out, y)
                print(f"Epoch: {epoch}, Loss: {np.mean(loss)}")

    def predict(self, x):
        self.forward(x)
        return self.a_out