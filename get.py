import os
import numpy as np

path = 'tokenised.txt'

chars = []

with open(os, 'r') as f:
    for char in f.read():
        char.append(char)

    with open('ai-learning\RNN\data\tokenised.txt', 'w') as s:
        s.write(str(chars))



