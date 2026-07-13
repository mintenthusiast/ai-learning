import numpy as np
import torchvision

class CNN:
    def __init__(self, stride, learning_rate, epochs, hidden_size):
        self.stride = stride
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.kernel = np.array([[1, 1, 1], 
                               [1, -8, 1],
                               [1, 1, 1]])
        
        self.kernel_x, self.kernel_y = 3, 3

        self.h1 = self.weight_init(, hidden_size)
        self.out = self.weight_init()
        
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
    
    def d_reLu(self, x):
        return np.where(x > 0, 1, 0)

    def weight_init(self, in_size, out_size):
        std = np.sqrt(2.0 / in_size)
        weights = np.random.randn(in_size, out_size) * std
        return weights




