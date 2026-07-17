import numpy as np
from collections import Counter
import os
import re

class RNN:
    def __init__(self, learning_rate, corpus_path, seq_len, no_of_seq, h1_size=512, h2_size=256, epochs=50, max_vocab=6000):
        self.lr = learning_rate
        self.corpus_path = corpus_path
        self.seq_len = seq_len
        self.no_of_seq = no_of_seq
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.epochs = epochs
        self.max_vocab = max_vocab

        '''Corpus Variables'''
        self.corpus = []
        self.vocab = {}
        self.keys_to_vocab = {}

        self.vocab_size = 0
        self.corpus_size = 0

        self.load_corpus()  # loads corpus

        '''Weight + Bias Init'''
        self.out_size = self.vocab_size

        self.w_hh1 = self.weight_init((self.h1_size, self.h1_size))  # (h1_size, h1_size)
        self.w_hh2 = self.weight_init((self.h2_size, self.h2_size))  # (h2_size, h2_size)

        self.w_h1 = self.weight_init((self.vocab_size, self.h1_size))   # (vocab_size, h1_size)
        self.w_h2 = self.weight_init((self.h1_size, self.h2_size))      # (h1_size, h2_size)
        self.w_o = self.weight_init((self.h2_size, self.vocab_size))    # (h2_size, vocab_size)

        self.b_h1 = np.zeros((1, self.h1_size))
        self.b_h2 = np.zeros((1, self.h2_size))
        self.b_o = np.zeros((1, self.out_size))

    def weight_init(self, dims):
        std = np.sqrt(2.0 / dims[0])
        weights = np.random.randn(dims[0], dims[1]) * std
        return weights

    def load_corpus(self):
        with open(self.corpus_path, 'r') as f:  # read file
            content = f.read()

        words = re.findall(r"[a-zA-Z0-9']+", content.lower())
        counts = Counter(words)
        self.most_frequent_words = counts

        self.vocab = {"<unk>": 0}
        self.keys_to_vocab = {0: "<unk>"}

        # counts.most_common returns (word, count) tuples
        for i, (word, _) in enumerate(counts.most_common(self.max_vocab - 1), start=1):
            self.vocab[word] = i
            self.keys_to_vocab[i] = word

        self.vocab_size = len(self.vocab)
        self.corpus_size = len(words)

        self.corpus = np.array([self.encode(w if w in self.vocab else "<unk>") for w in words])  # (corpus_size, vocab_size)


    def get_batch(self):
        x = np.zeros((self.no_of_seq, self.seq_len, self.vocab_size))
        y = np.zeros_like(x)

        random_indexes = np.random.randint(0, self.corpus_size - self.seq_len, self.no_of_seq)

        for i, j in enumerate(random_indexes):
            x[i] = self.corpus[j:j + self.seq_len]
            y[i] = self.corpus[j + 1:j + self.seq_len + 1]

        return x, y

    def forward(self, x, state_h1=None, state_h2=None):
        # x: (batch, seq_len, vocab_size)
        x = x.transpose(1, 0, 2)  # (seq_len, batch, vocab_size)
        seq_len, batch_size = x.shape[0], x.shape[1]

        if state_h1 is None:
            state_h1 = np.zeros((batch_size, self.h1_size))

        if state_h2 is None:
            state_h2 = np.zeros((batch_size, self.h2_size))

        h1_states = np.zeros((seq_len, batch_size, self.h1_size))
        h1_z = np.zeros_like(h1_states)

        for t in range(seq_len):
            X = x[t]  # (batch, vocab_size)
            state_h1, z = self.update_state(X, self.tanh, state_h1,
                                            self.w_h1, self.w_hh1, self.b_h1)
            h1_z[t] = z
            h1_states[t] = state_h1

        h2_states = np.zeros((seq_len, batch_size, self.h2_size))
        h2_z = np.zeros_like(h2_states)

        for t in range(seq_len):
            X = h1_states[t]  # (batch, h1_size)
            state_h2, z = self.update_state(X, self.tanh, state_h2,
                                            self.w_h2, self.w_hh2, self.b_h2)
            h2_z[t] = z
            h2_states[t] = state_h2

        logits = np.dot(h2_states, self.w_o) + self.b_o  # (seq_len, batch, vocab_size)

        return x, h1_states, h1_z, h2_states, h2_z, logits

    def backward(self, x, y, h1_states, h1_z, h2_states, h2_z, logits):
        # x:   (seq_len, no_of_seq, vocab_size)
        # y:   (no_of_seq, seq_len, vocab_size)
        y = y.transpose(1, 0, 2)  # (seq_len, no_of_seq, vocab_size)

        # Output layer gradients
        grad_logits = self.softmax(logits) - y  # (seq_len, no_of_seq, vocab_size)

        dW_o = np.dot(h2_states.reshape(-1, self.h2_size).T,
                      grad_logits.reshape(-1, self.out_size))  # (h2_size, out_size)
        dB_o = grad_logits.sum(axis=(0, 1), keepdims=False).reshape(1, -1)  # (1, out_size)

        # Backprop into h2 states
        grad_h2 = np.dot(grad_logits, self.w_o.T)  # (seq_len, no_of_seq, h2_size)

        dW_h2 = np.zeros_like(self.w_h2)
        dW_hh2 = np.zeros_like(self.w_hh2)
        dB_h2 = np.zeros_like(self.b_h2)

        grad_h2_next = np.zeros((self.no_of_seq, self.h2_size))
        grad_h1_from_h2 = np.zeros((self.seq_len, self.no_of_seq, self.h1_size))

        for t in reversed(range(self.seq_len)):
            grad_h2_total = grad_h2[t] + grad_h2_next  # (no_of_seq, h2_size)
            grad_z_h2 = grad_h2_total * (1 - np.tanh(h2_z[t]) ** 2)

            grad_h2_next = np.dot(grad_z_h2, self.w_hh2.T)
            grad_h1_from_h2[t] = np.dot(grad_z_h2, self.w_h2.T)  # gradient w.r.t h1_states[t]

            if t > 0:
                h_prev = h2_states[t - 1]
            else:
                h_prev = np.zeros((self.no_of_seq, self.h2_size))

            X_t = h1_states[t]
            dW_h2 += np.dot(X_t.T, grad_z_h2)
            dW_hh2 += np.dot(h_prev.T, grad_z_h2)
            dB_h2 += grad_z_h2.sum(axis=0, keepdims=False).reshape(1, -1)

        # Backprop through h1 layer
        dW_h1 = np.zeros_like(self.w_h1)
        dW_hh1 = np.zeros_like(self.w_hh1)
        dB_h1 = np.zeros_like(self.b_h1)

        grad_h1_next = np.zeros((self.no_of_seq, self.h1_size))

        for t in reversed(range(self.seq_len)):
            grad_h1_total = grad_h1_from_h2[t] + grad_h1_next  # (no_of_seq, h1_size)
            grad_z_h1 = grad_h1_total * (1 - np.tanh(h1_z[t]) ** 2)

            grad_h1_next = np.dot(grad_z_h1, self.w_hh1.T)

            if t > 0:
                h_prev = h1_states[t - 1]
            else:
                h_prev = np.zeros((self.no_of_seq, self.h1_size))

            X_t = x[t]
            dW_h1 += np.dot(X_t.T, grad_z_h1)
            dW_hh1 += np.dot(h_prev.T, grad_z_h1)
            dB_h1 += grad_z_h1.sum(axis=0, keepdims=False).reshape(1, -1)

        # Gradient clipping for stability
        for grad in [dW_o, dB_o, dW_h1, dW_hh1, dB_h1, dW_h2, dW_hh2, dB_h2]:
            np.clip(grad, -1.0, 1.0, out=grad)

        # Parameter updates
        self.w_o -= self.lr * dW_o
        self.w_h1 -= self.lr * dW_h1
        self.w_h2 -= self.lr * dW_h2
        self.w_hh1 -= self.lr * dW_hh1
        self.w_hh2 -= self.lr * dW_hh2

        self.b_o -= self.lr * dB_o
        self.b_h1 -= self.lr * dB_h1
        self.b_h2 -= self.lr * dB_h2

    def fit(self):
        for epoch in range(self.epochs):
            w1_prev, w2_prev = self.w_h1.copy(), self.w_h2.copy()

            x, y = self.get_batch()
            x_t, h1_states, h1_z, h2_states, h2_z, logits = self.forward(x)
            self.backward(x_t, y, h1_states=h1_states, h2_states=h2_states,
                          h1_z=h1_z, h2_z=h2_z, logits=logits)

            if epoch % 5 == 0:
                loss = self.cross_entropy_loss(self.softmax(logits), y.transpose(1, 0, 2))
                print(f"h1 change: {np.linalg.norm(self.w_h1 - w1_prev)}, "
                  f"h2_change {np.linalg.norm(self.w_h2 - w2_prev)}")
                print(f"Epoch: {epoch} Loss: {loss}")

                self.lr *= 0.95

    def generate(self, seed, l=100, temp=1.2):
        words = re.findall(r"[a-zA-Z0-9']+", seed.lower())
        encoded = np.zeros((1, len(words), self.vocab_size))

        for i, word in enumerate(words):
            encoded[0, i] = self.encode(word)

        _, h1_states, _, h2_states, _, logits = self.forward(encoded)
        state_h1 = h1_states[-1]  # (1, h1_size)
        state_h2 = h2_states[-1]  # (1, h2_size)

        generated = list(words)

        for _ in range(l):
            probs = self.softmax(logits / temp)  # (seq_len, 1, vocab_size)
            next_index = np.random.choice(self.vocab_size, p=probs[-1, 0])
            next_word = self.keys_to_vocab[next_index]
            generated.append(next_word)

            next_input = self.encode(next_word).reshape(1, 1, self.vocab_size)
            _, h1_states, _, h2_states, _, logits = self.forward(
                next_input, state_h1=state_h1, state_h2=state_h2)
            state_h1 = h1_states[-1]
            state_h2 = h2_states[-1]

        return ' '.join(generated)

    def top(self, b=500):
        if b > self.max_vocab:
            return "Top X greater than max vocab recorded."
        
        return self.most_frequent_words.most_common(b)

    def update_state(self, x, function, state, w_h, w_hh, b_h):
        a = np.dot(x, w_h) + np.dot(state, w_hh) + b_h
        return function(a), a

    def encode(self, word):
        out = np.zeros(self.vocab_size)
        idx = self.vocab.get(word, self.vocab["<unk>"])
        out[idx] = 1.0
        return out

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.maximum(0, x).astype(float)

    def d_relu(self, x):
        return np.where(x > 0, 1, 0)

    def cross_entropy_loss(self, p, y):
        # p, y: (seq_len, no_of_seq, vocab_size)
        loss = -np.sum(y * np.log(p + 1e-15)) / (self.seq_len * self.no_of_seq)
        return loss

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


path = os.path.join('RNN', 'data', 'input.txt')

rnn = RNN(learning_rate=0.0001, corpus_path=path, seq_len=16, no_of_seq=256,
          h1_size=512, h2_size=256, epochs=500)

rnn.fit()
print("\n")
rnn.top(b=20)
print("\n")
print(rnn.generate("This is", l=50, temp=1.7))
print("\n")
print(rnn.generate("This is a strange repose, to be asleep, With eyes wide open;", l=50, temp=0.6))
print("\n")
print(rnn.vocab, "\n", )
