import numpy as np
from numpy.random import uniform as u
from math import sqrt
from _collections import defaultdict

class NeuralNetwork:

    # lr        =   Learning Rate
    # n_sizes   =   Size of each layer in neurons, passed as a list. Includes the
    #               number of inputs. E.g.: [3, 4, 3, 2, 1]
    # w, b      =   Arrays of parameters for each layer, i.e: b[3] corresponds to b for layer 3.
    # caches     =   Used to store A and Z caches as a list of tuples.
    def __init__(self, n_sizes, lr, train_data, g=None, gprime=None, epochs=None):
        assert n_sizes[0] == len(train_data[0])
        self.n = len(n_sizes)

        # Xavier/Glorot Initialisation
        self.params = {}
        for l in range(1, self.n):
            xgb = sqrt(6) / sqrt(n_sizes[l] + n_sizes[l - 1])
            self.params["W"+str(l)] = np.array([u(-xgb, xgb, n_sizes[l - 1]) for _ in range(n_sizes[l])])
            self.params["b"+str(l)] = np.zeros((n_sizes[l], 1))
        self.lr = lr
        self.X = np.array(train_data[0])
        self.Y = np.array(train_data[1])
        self.caches = [[]]
        self.m = len(self.Y)
        self.final_activation = np.empty(np.shape(self.Y))
        self.acc = 0
        self.gradients = {}
        if epochs is None:
            self.epochs = 100
        else:
            self.epochs = epochs

    def __accuracy(self):
        self.acc = np.sum((self.final_activation - self.Y) ** 2) / self.m

    @staticmethod
    def __forward_prop_calcs(A_prev, W, b, activation):
        Z = np.dot(W, A_prev) + b
        return activation(Z), ((A_prev, W, b), Z) # Returns A and Cache

    def __forward_prop(self):
        caches = []
        A = self.X
        for l in range(1, self.n-1):
            W = self.params["W" + str(l)]
            b = self.params["b" + str(l)]
            A_prev = A
            tanh = lambda x: np.tanh(x)
            A, cache = self.__forward_prop_calcs(A_prev, W, b, tanh)
            caches.append(cache)
        Wn = self.params["W" + str(self.n-1)]
        bn = self.params["b" + str(self.n-1)]
        softmax = lambda x: np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum(axis=0)
        self.final_activation, finalC = self.__forward_prop_calcs(A, Wn, bn, softmax)
        caches.append(finalC)
        self.caches = caches

    def __backward_prop_calcs(self, cachen, layern, backwards_func, dA):
        lcache, acache = self.caches[cachen]
        dZ = dA * backwards_func(acache)
        A, W, b = lcache
        m = A.shape[1]
        dW = np.dot(dZ, (A.T)) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA = np.dot(W.T, dZ)
        self.gradients["dA" + str(layern-1)] = dA
        self.gradients["dW" + str(layern)] = dW
        self.gradients["db" + str(layern)] = db

    @staticmethod
    def softmax_deriv(Z):
        softmax_probs = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        deriv = softmax_probs * (1 - softmax_probs)
        return deriv

    def __gradient_descent(self):
        final_activation_derivative = (self.final_activation - self.Y) / self.m
        lcache, acache = self.caches[-1]
        dZ = final_activation_derivative * self.softmax_deriv(acache)
        A, W, b = lcache
        m = A.shape[1]
        dW = np.dot(dZ, (A.T)) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA = np.dot(W.T, dZ)
        self.gradients["dW" + str(self.n-1)] = dW
        self.gradients["db" + str(self.n-1)] = db
        tanh_deriv = lambda x : 1 - np.tanh(x)**2                   # 1ST ITERATION
        for l in range(len(self.caches)-2, -1, -1):                 # We want to start with our first unaccessed cache.
            lcache, acache = self.caches[l]                         # We extract the separate caches
            dZ = dA * tanh_deriv(acache)                            # We use dA3 and Z3 to calculate dZ3
            A, W, b = lcache                                        # We retrieve A2, W3 and b3.
            m = A.shape[1]                                          # Same as before.
            dW = np.dot(dZ, (A.T)) / m                              # We calculate dW3
            db = np.sum(dZ, axis=1, keepdims=True) / m              # We calculate db3
            dA = np.dot(W.T, dZ)                                    # We calculate dA2
            self.gradients["dW" + str(l+1)] = dW
            self.gradients["db" + str(l+1)] = db

        for l in range(self.n-1):
            self.params["W" + str(l+1)] = self.params["W" + str(l+1)] - self.gradients["dW" + str(l+1)] * self.lr
            self.params["b" + str(l+1)] = self.params["b" + str(l+1)] - self.gradients["db" + str(l+1)] * self.lr

    def train(self):
        for epoch in range(self.epochs):
            self.__forward_prop()
            self.__accuracy()
            self.__gradient_descent()
            print(f"Epoch {epoch + 1}: Cost = {self.acc}")