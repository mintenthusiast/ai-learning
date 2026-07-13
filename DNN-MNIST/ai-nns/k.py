import numpy as np

a1 = np.array([[1, 2, 3, 4, 5], 
               [2, 3, 4, 5, 6],
               [3, 4, 5, 6, 7],
               [4, 5, 6, 7, 8],
               [5, 6, 7, 8, 9]])

a2 = np.array([[1, 2, 3],
               [2, 3, 4],
               [3, 4, 5]])

col, row = 0, 0

a3 = np.zeros((3, 3))
section = a1[col:col+3, row:row+3]
sum = np.sum(section * a2)
