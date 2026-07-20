from collections import Counter
import numpy as np
import re
import os

class Layer:
    def __init__(self, dims, function, state, recurrent=True):
        self.dims = dims
        self.function = function
        
        self.state = state
        self.weight_matrix = self._weight_init(dims)
        if recurrent:
            self.internal_weight_matrix = self._weight_init((dims[1], dims[1]))

        self.biases = np.zeros((1, dims[1]))
    
    def _weight_init(self, dims):
        std = np.sqrt(1.0 / dims[0])
        weights = np.random.randn(dims[0], dims[1]) * std
        return weights
    
class RNN:
    def __init__(self, lr=0.001, hidden_layers=2, h_sizes=[512, 256], load=False, max_vocab=6000, seq_length=32, seq_number=64, path=""):
        self.lr = lr
        self.max_vocab = max_vocab
        self.seq_length = seq_length
        self.seq_number = seq_number
        self.no_layers = hidden_layers + 2

        self.corpus = []
        self.vocab = []
        self.words_to_idx = {}

        self._load_corpus(path)

        self.corpus_size = len(self.corpus)
        self.vocab_size = len(self.vocab)

        if len(h_sizes) != hidden_layers:
            raise ValueError

        self.layer_sizes = [self.vocab_size]

        for x in h_sizes:
            self.layer_sizes.append(x)

        self.layer_sizes.append(self.vocab_size)
        
        self.layers = []
        self.forward_cache = []
        self.delta_cache = []

        for i in range(1, self.no_layers - 1):
            dims = (self.layer_sizes[i - 1], self.layer_sizes[i])
            newLayer = Layer(dims=dims, function=self._tanh, state=np.zeros((self.seq_number, dims[1])))
            self.layers.append(newLayer)

        dims = (self.layer_sizes[-2], self.layer_sizes[-1])

        newLayer = Layer(dims=dims, function=None,
                          state=None, recurrent=False)
        
        self.layers.append(newLayer)
        
    def forward(self, x, reset_state=True):
        x = x.transpose(1, 0, 2)
        seq_len = x.shape[0]

        if reset_state:
            for layer in self.layers[:-1]:
                layer.state = np.zeros((x.shape[1], layer.dims[1]))

        self.forward_cache = [(x, None, None)]
        a_prev = x

        for layer in self.layers[:-1]:
            states = np.zeros((seq_len, x.shape[1], layer.dims[1]))
            z_i = np.zeros_like(states)

            for t in range(seq_len):
                X_t = a_prev[t]
                layer.state, z = self._update_state(X_t, layer.function, layer.state, layer.weight_matrix, 
                                                    layer.internal_weight_matrix, layer.biases)
                z_i[t] = z
                states[t] = layer.state
            
            self.forward_cache.append((a_prev, z_i, states))
            a_prev = states
        
        logits = np.dot(a_prev, self.layers[-1].weight_matrix) + self.layers[-1].biases
        self.forward_cache.append((a_prev, logits, self._softmax(logits)))
        return logits
    
    def backward(self, x, y):
        y = y.transpose(1, 0, 2)
        x = x.transpose(1, 0, 2)

        seq_len = x.shape[0]
        batch_size = x.shape[1]
        m = batch_size

        grad_logits = (self.forward_cache[-1][-1] - y) / m

        h_last = self.forward_cache[-1][0]

        dW_o = np.dot(h_last.reshape(-1, h_last.shape[-1]).T, grad_logits.reshape(-1, grad_logits.shape[-1])) # size of previous layer, vocab size
        db_o = np.sum(grad_logits, axis=(0, 1), keepdims=False).reshape(1, -1) # 1, vocab size

        for grad in [dW_o, db_o]:
            np.clip(grad, -1.0, 1.0, out=grad)

        self.layers[-1].weight_matrix -= self.lr * dW_o
        self.layers[-1].biases -= self.lr * db_o

        self.delta_cache.append((dW_o, None, db_o))

        grad_h_last = np.dot(grad_logits.reshape(-1, self.vocab_size), self.layers[-1].weight_matrix.T)
        grad_h_last = grad_h_last.reshape(seq_len, batch_size, -1)

        for i in reversed(range(len(self.layers) - 1)):
            z = self.forward_cache[i + 1][1]
            a_prev = self.forward_cache[i + 1][0]
            states = self.forward_cache[i + 1][-1]

            layer = self.layers[i]

            dW_h = np.zeros_like(layer.weight_matrix)
            dW_hh = np.zeros_like(layer.internal_weight_matrix)
            db_h = np.zeros_like(layer.biases)

            grad_h_next = np.zeros((batch_size, layer.dims[1]))
            grad_input_layer = np.zeros((seq_len, batch_size, layer.dims[0]))

            for t in reversed(range(seq_len)):
                h_total = grad_h_last[t] + grad_h_next
                grad_z = h_total * self._d_tanh(z[t])

                grad_h_next = np.dot(grad_z, layer.internal_weight_matrix.T)
                grad_input_layer[t] = np.dot(grad_z, layer.weight_matrix.T)

                if t > 0:
                    h_prev = states[t - 1]
                else:
                    h_prev = np.zeros((batch_size, layer.dims[1]))
            
                dW_h += np.dot(a_prev[t].T, grad_z)
                dW_hh += np.dot(h_prev.T, grad_z)
                db_h += np.sum(grad_z, axis=0, keepdims=False).reshape(1, -1) # 1, layer bias size
            
            grad_h_last = grad_input_layer

            for grad in [dW_h, dW_hh, db_h]:
                np.clip(grad, -1.0, 1.0, out=grad)

            self.delta_cache.append((dW_h, dW_hh, db_h))

            layer.weight_matrix -= self.lr * dW_h
            layer.internal_weight_matrix -= self.lr * dW_hh
            layer.biases -= self.lr * db_h
    
    def fit(self, epochs):
        for epoch in range(epochs):
            x, y = self._get_batch()
            p = self.forward(x)
            p = self._softmax(p)

            self.backward(x, y)

            y_t = y.transpose(1, 0, 2)

            loss = np.mean(self.cross_entropy_loss(p, y_t))
            acc = np.mean(np.argmax(p, axis=-1) == np.argmax(y_t, axis=-1))

            print(f"Epoch: {epoch}, Loss: {loss:.8f}, Accuracy: {acc:.8f}")
            self.delta_cache = []
        
        print("done training.")

    def generate(self, seed, l=100, temp=1.2):
        words = re.findall(r"[a-zA-Z0-9']+", seed.lower())
        if not words:
            return ""

        encoded = np.zeros((1, len(words), self.vocab_size))
        for i, word in enumerate(words):
            encoded[0, i] = self.encode(word)

        self.forward(encoded)  # seed forward; resets hidden state

        generated = list(words)

        for _ in range(l):
            logits = self.forward_cache[-1][1]  # (seq_len, 1, vocab_size)
            probs = self._softmax(logits[-1] / temp).ravel()  # (vocab_size,)
            next_index = np.random.choice(self.vocab_size, p=probs)
            next_word = self.vocab[next_index]
            generated.append(next_word)

            next_input = self.encode(next_word).reshape(1, 1, self.vocab_size)
            self.forward(next_input, reset_state=False)  # carry hidden state forward

        return ' '.join(generated)
            
    def _load_corpus(self, path):
        with open(path, 'r') as f:
            content = f.read()

        words = re.findall(r"[a-zA-Z0-9']+", content.lower())
        counts = Counter(words)

        top_words = counts.most_common(self.max_vocab - 1)

        self.vocab = ["<unk>"] + [w for w, _ in top_words]
        self.words_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.corpus = words

    def _get_batch(self):
        x_batch = []
        y_batch = []

        idx = np.random.randint(0, self.corpus_size - self.seq_length, self.seq_number)
        
        for i in idx:
            tmp = self.corpus[i : i + self.seq_length + 1]
            for j in range(len(tmp)):
                tmp[j] = self.encode(tmp[j])
            
            x_batch.append(tmp[:self.seq_length])
            y_batch.append(tmp[1:self.seq_length + 1])

        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
    
        return x_batch, y_batch

    def encode(self, x):
        encoded = np.zeros((self.vocab_size))
        idx = self.words_to_idx.get(x, self.words_to_idx["<unk>"])
        encoded[idx] = 1.0
        return encoded
    
    def decode(self, x):
        for i in range(len(x)):
            if x[i] == 1.0:
                return self.vocab[i]
            
        return
    
    def _tanh(self, x):
        return np.tanh(x)
    
    def _d_tanh(self, x):
        return 1 - np.tanh(x) ** 2

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _update_state(self, x, function, state, w_h, w_hh, b_h):
        a = np.dot(x, w_h) + np.dot(state, w_hh) + b_h
        return function(a), a
    
    def cross_entropy_loss(self, p, y):
        loss = -np.sum(y * np.log(p + 1e-15)) / (self.seq_length * self.seq_number)
        return loss
    
script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.normpath(os.path.join(script_dir, '..', 'RNN', 'data', 'input.txt'))
    
rnn = RNN(lr=0.01, hidden_layers=4, h_sizes=[512, 256, 128, 64], load=False, max_vocab=2000,
          seq_length=32, seq_number=64, path=path)

rnn.fit(epochs=50)
print(rnn.generate("There is", l=50, temp=1.2))