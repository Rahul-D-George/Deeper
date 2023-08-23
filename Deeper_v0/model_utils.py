import numpy as np
from numpy.random import uniform as u
from math import sqrt


class NeuralNetwork:
    # lr        =   Learning Rate
    # n_sizes   =   Size of each layer in neurons, passed as a list. Includes the
    #               number of inputs. E.g.: [3, 4, 3, 2, 1]
    # w, b      =   Arrays of parameters for each layer, i.e: b[3] corresponds to b for layer 3.
    # cache     =   Used to store Z[l] and A[l] in a (n, 2) array. DOES NOT INCLUDE INPUT FROM TRAINING SET
    def __init__(self, n_sizes, lr, train_data, g=None):
        assert n_sizes[0] == len(train_data[0][0])
        self.layers = len(n_sizes)

        # True random initialisation
        # self.w = [np.array([[rd() for column in range(n_sizes[i - 1])]
        #           for row in range(n_sizes[i])]) for i in range(1, self.layers)]

        # Xavier/Glorot Initialisation
        self.w = []
        for l in range(1, self.layers):
            xgb = sqrt(6)/sqrt(n_sizes[l] + n_sizes[l-1])
            self.w.append(np.array([[u(-xgb, xgb, n_sizes[l-1])] for row in range(n_sizes[l])]))

        self.b = [np.array([[0] for row in range(n_sizes[i])]) for i in range(1, self.layers)]
        self.lr = lr
        self.X = np.array(train_data[0])
        self.Y = train_data[1]
        self.cache = [[] for layer in range(self.layers)]
        if g is None:
            self.g = [np.tanh for layer in range(self.layers)]
        else:
            self.g = g

    def forward_prop(self):
        a = self.X
        for l in range(1, self.layers):
            z = np.dot(self.w[l], a)
            a = self.g[l](z)
            self.cache[l].append([z, a])