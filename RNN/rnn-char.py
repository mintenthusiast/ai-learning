import numpy as np
from collections import defaultdict
from collections import Counter
import os

class RNN:
    def __init__(self, learning_rate, hidden_size, corpus_path, seq_len=26, epochs=50, batch_size=5):
        self.lr = learning_rate
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.corpus_path = corpus_path

        self.seen = defaultdict()
        self.keys = defaultdict()

        self.a_h = 0.0
        self.a_o = 0.0

        self.corpus = []

        self.load_corpus(self.corpus_path)
        self.out_size = len(self.seen)

        self.w_h = self.weight_init((self.out_size, hidden_size)) # (vocab_size, hidden_size)
        self.w_hh = self.weight_init((hidden_size, hidden_size)) # (hidden_size, hidden_size)
        self.w_o = self.weight_init((hidden_size, self.out_size)) # (hidden_size, vocab_size)

        self.b_h = self.weight_init((1, hidden_size)) # (1, hidden_size)
        self.b_o = self.weight_init((1, self.out_size)) # (1, vocab_size)

    def weight_init(self, dims):
        std = np.sqrt(2.0 / dims[0])
        weights = np.random.randn(dims[0], dims[1]) * std
        return weights
    
    def update_state(self, state, function, x): # returns: 
        a = np.dot(x, self.w_h) + np.dot(state, self.w_hh) + self.b_h
        return function(a), a
    
    def tanh(self, x):
        return np.tanh(x)
    
    def relu(self, x):
        return np.maximum(0, x).astype(float)

    def d_relu(self, x):
        return np.where(x > 0, 1, 0)
    
    def cross_entropy_loss(self, p, y):
        loss = -np.sum(y * np.log(p + 1e-15), axis=1)
        return loss
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def encode(self, char):
        v = np.zeros(len(self.seen))
        v[self.seen[char]] = 1.0
   
        return v

    def load_corpus(self, path): # should return (batch_size, seq_len, vocab_size)
        self.corpus = []
        self.seen = defaultdict()

        with open(f'{path}', 'r') as f:
            curr = 0
            for char in f.read():
                if char in ('\n', '\t'):
                    continue
                elif char not in self.seen:
                    self.seen[char] = curr
                    self.keys[curr] = char
                    curr += 1

                self.corpus.append(char)
            
            for i in range(len(self.corpus)):
                encoded_char = self.encode(self.corpus[i])
                self.corpus[i] = encoded_char # (no. chars, len(seen))

    def get_batch(self): # should return (batch_size, seq_len, vocab_size)
        x = np.zeros((self.batch_size, self.seq_len, len(self.seen)))
        y = np.zeros_like(x)
        tmp = np.random.randint(0, len(self.corpus) - self.seq_len, self.batch_size)

        for i, j in enumerate(tmp):
            x[i] = self.corpus[j:j + self.seq_len]
            y[i] = self.corpus[j + 1:j + self.seq_len + 1]

        return x, y

    def forward(self, x, l, state=None):
        # x: (batch_size, seq_len, vocab_size)
        x = x.transpose(1, 0, 2)
        if state is None:
            state = np.zeros((x.shape[1], self.hidden_size)) # (batch_size, hidden_size)
        else:
            state, = state
        
        out = np.zeros((l, self.batch_size, self.hidden_size)) # (seq_len, batch_size, hidden_size)
        pre = np.zeros_like(out)

        for t in range(l):
            X = x[t] # (batch_size, vocab_size)
            state, pre_a = self.update_state(state, self.tanh, X) 
            pre[t] = pre_a
            out[t] = state

        logits = np.dot(out, self.w_o) + self.b_o # (seq_len, batch_size, vocab_size)

        return out, pre, logits
    
    def backward(self, x, y, out, pre, logits):
        x = x.transpose(1, 0, 2)
        y = y.transpose(1, 0, 2)

        grad_logits = self.softmax(logits) - y
        dW_o = np.dot(out.reshape(-1, self.hidden_size).T, grad_logits.reshape(-1, self.out_size))
        db_o = grad_logits.sum(axis=(0, 1), keepdims=True).reshape(1, self.out_size)

        grad_h = np.dot(grad_logits, self.w_o.T)

        dW_h = np.zeros_like(self.w_h)
        dW_hh = np.zeros_like(self.w_hh)
        db_h = np.zeros_like(self.b_h)

        grad_h_next = np.zeros((self.batch_size, self.hidden_size))
    
        for t in reversed(range(self.seq_len)):
            grad_h_total = grad_h[t] + grad_h_next
            grad_z = grad_h_total * (1 - np.tanh(pre[t]) ** 2)

            grad_h_next = np.dot(grad_z, self.w_hh.T)

            if t > 0:
                h_prev = out[t - 1]
            else:
                h_prev = np.zeros((self.batch_size, self.hidden_size))

            X_t = x[t]

            dW_h += np.dot(X_t.T, grad_z)
            dW_hh += np.dot(h_prev.T, grad_z)
            db_h += grad_z.sum(axis=0, keepdims=True)
        
        for grad in [dW_o, dW_h, dW_hh, db_o, db_h]:
            np.clip(grad, -0.5, 0.5, out=grad)

        self.w_o -= self.lr * dW_o
        self.b_o -= self.lr * db_o
        self.w_h -= self.lr * dW_h
        self.w_hh -= self.lr * dW_hh
        self.b_h -= self.lr * db_h

    def fit(self):
        for epoch in range(self.epochs):
            x, y = self.get_batch()
            out, pre, logits = self.forward(x, self.seq_len)
            w_before = self.w_h.copy()
            self.backward(x, y, out, pre, logits)
            print("weight change:", np.linalg.norm(self.w_h - w_before))
            if epoch % 1 == 0:
                loss = -np.sum(y.transpose(1, 0, 2) * np.log(self.softmax(logits) + 1e-15)) / (self.seq_len * self.batch_size)
                print(f"epoch {epoch}, loss: {loss:.4f}")

    def test(self):
        x, y = self.get_batch()
        _, _, a = self.forward(x)
        return a
    
    def generate(self, seed, length=100, temperature=0.7):
        encoded = np.zeros((1, len(seed), len(self.seen)))

        for i, char in enumerate(seed):
            encoded[0, i] = self.encode(char)
        out, _, _ = self.forward(encoded, len(seed))
        state = out[-1]

        generated = seed

        for _ in range(length):
            logits = np.dot(state, self.w_o) + self.b_o
            probs = self.softmax(logits / temperature)
            # print(probs)
            # print(np.max(np.abs(logits)))

            next_index = np.random.choice(len(self.seen), p=probs[0])
            next_char = self.keys[next_index]
            generated += next_char

            next_input = self.encode(next_char).reshape(1, -1)
            state, _ = self.update_state(state, self.tanh, next_input)
        
        return generated
    
    def count(self):
        with open(self.corpus_path) as f:
            text = f.read()
        print(Counter(text).most_common(10))
    
path = os.path.join('RNN', 'data', 'input.txt')
rnn = RNN(learning_rate=0.00001, epochs=200, hidden_size=256, seq_len=32, corpus_path=path, batch_size=256)
rnn.fit()

print(repr(rnn.generate("tho", length=100, temperature=1.2)))
print(rnn.generate("You are gay, ", length=100, temperature=1.2))
print(rnn.generate("In nineteen-thirty", length=100, temperature=1.5))

rnn.count()
