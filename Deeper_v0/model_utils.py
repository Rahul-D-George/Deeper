import numpy as np
from numpy.random import uniform as u
from math import sqrt


class NeuralNetwork:
    # lr        =   Learning Rate
    # n_sizes   =   Size of each layer in neurons, passed as a list. Includes the
    #               number of inputs. E.g.: [3, 4, 3, 2, 1]
    # w, b      =   Arrays of parameters for each layer, i.e: b[3] corresponds to b for layer 3.
    # cache     =   Used to store Z[l] and A[l] in a (n, 2) array. DOES NOT INCLUDE A0.
    def __init__(self, n_sizes, lr, train_data, g=None, gprime=None):
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
        self.cache[0][1] = self.X
        self.m = len(self.Y)
        self.cost = 0
        if g is None:
            self.g = [np.tanh for _ in range(self.n - 1)]
            self.g.append(np.identity_)
            self.gprime = [lambda x : 1 - np.tanh(x)**2 for _ in range(self.n - 1)]
            self.gprime.append(np.identity_)
        else:
            self.g = g
            self.gprime = gprime

    def __mse_cost(self):
        self.cost = np.sum((self.cache[-1][1] - self.Y)**2)/self.m

    def __forward_prop(self):
        A = self.X
        for l in range(1, self.n):
            Z = np.dot(self.W[l], A)
            A = self.g[l](Z)
            self.cache[l].append([Z, A])

    def __gradient_descent(self):
        dA = 2 * (self.cache[-1][1] - self.Y) # Derivative of MSE function is literally just 2 * (Y - Yhat)
        for l in range(self.n - 1, -1, -1): # Backprop Step :D
            dZ = dA * self.gprime[l](self.cache[l][0])
            dW = (1/self.m) * np.dot(dZ, self.cache[l-1][1].T)
            dB = (1/self.m) * np.sum(dZ, axis=1, keepdims=True)
            self.W[l] = self.W[l] - (self.lr * dW) # Updating weight matrix
            self.b[l] = self.b[l] - (self.lr * dB) # Updating biases
            dA = np.dot(self.W[l].T, dZ)

