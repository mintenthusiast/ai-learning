import numpy as np
from collections import defaultdict

class RNN:
    def __init__(self, learning_rate, epochs, hidden_size, out_size):
        self.lr = learning_rate
        self.epochs = epochs
        self.hidden_size = hidden_size

        self.w_h = self.weight_init((in_size, hidden_size)) # (batch_size, hidden_size)
        self.w_hh = self.weight_init((hidden_size, hidden_size)) # (hidden_size, hidden_size)
        self.w_o = self.weight_init((hidden_size, out_size)) # (hidden_size, out_size)

        self.b_h = self.weight_init((1, hidden_size)) # (1, hidden_size)
        self.b_o = self.weight_init((1, out_size)) # (1, out_size)

        self.seen = defaultdict()
        self.keys = defaultdict()

        self.corpus = []

    def weight_init(self, dims):
        std = np.sqrt(2.0 / dims[0])
        weights = np.random.randn(dims[0], dims[1]) * std
        return weights
    
    def update_state(self, state, function, x):
        return function(np.dot(x, self.w_h) + np.dot(state, self.w_hh) + self.b_h)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def relu(self, x):
        return np.maximum(0, x).astype(float)

    def d_relu(self, x):
        return np.where(x > 0, 1, 0)


    def load_corpus(self, path):
        with open(f'{path}', 'r') as f:
            curr = 0
            for char in f.read():
                if char not in self.seen:
                    self.seen[char] = curr
                    self.keys[curr] = char
                    curr += 1

                self.corpus.append(self.seen[char])

    def forward(self, x, state=None):
        if state is None:
            state = np.zeros((x.shape[1], self.hidden_size)) # (batch_size, hidden_size)
        else:
            state, = state
        
        out = []

        for X in x:
            state = self.update_state(state, self.tanh, X)
            out.append(state)

        return out, state

rnn = RNN(learning_rate=0.001, epochs=10, hidden_size=32, in_size=5, out_size=26)
