import numpy as np
from numpy.random import uniform as u
from math import sqrt


class NeuralNetwork:
    # lr        =   Learning Rate
    # n_sizes   =   Size of each layer in neurons, passed as a list. Includes the
    #               number of inputs. E.g.: [3, 4, 3, 2, 1]
    # w, b      =   Arrays of parameters for each layer, i.e: b[3] corresponds to b for layer 3.
    # cache     =   Used to store Z[l] and A[l] in a (n, 2) array. DOES NOT INCLUDE A0.
    def __init__(self, n_sizes, lr, train_data, g=None):
        assert n_sizes[0] == len(train_data[0][0])
        assert n_sizes[-1] == 1
        self.n = len(n_sizes)

        # True random initialisation
        # self.w = [np.array([[rd() for column in range(n_sizes[i - 1])]
        #           for row in range(n_sizes[i])]) for i in range(1, self.layers)]

        # Xavier/Glorot Initialisation
        self.W = []
        for l in range(1, self.n):
            xgb = sqrt(6)/sqrt(n_sizes[l] + n_sizes[l-1])
            self.W.append(np.array([[u(-xgb, xgb, n_sizes[l-1])] for _ in range(n_sizes[l])]))

        self.b = [np.array([[0] for _ in range(n_sizes[i])]) for i in range(1, self.n)]
        self.lr = lr
        self.X = np.array(train_data[0])
        self.Y = np.array(train_data[1])
        self.cache = [[] for _ in range(self.n)]
        self.m = len(self.Y)
        if g is None:
            self.g = [np.tanh for _ in range(self.n - 1)]
            self.g.append(np.identity_)
        else:
            self.g = g

    def forward_prop(self):
        A = self.X
        for l in range(1, self.n):
            Z = np.dot(self.W[l], A)
            A = self.g[l](Z)
            self.cache[l].append([Z, A])


    def gradient_descent(self):

    def back_prop(self):
        dA = dZ = 1  # We are using a linear activation function in the final layer.
        Atprev = self.cache[-1][1]
        for l in range(self.n - 1, 0, -1):
            dZ = dA *