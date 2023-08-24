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
        assert n_sizes[-1] == 30
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
        self.cost = 0
        self.gradients = {}
        if epochs is None:
            self.epochs = 100
        else:
            self.epochs = epochs
        # if g is None:
        #     self.g = [np.tanh for _ in range(self.n - 1)]
        #     self.g.append(lambda x: x)
        #     self.gprime = [self.tanh_derivative for _ in range(self.n - 1)]
        #     self.gprime.append(lambda x: 1)
        # else:
        #     self.g = g
        #     self.gprime = gprime

    #def __sparse_categorical_cross_entropy(self):
    #    print(np.shape(self.Y), np.shape(self.final_activation))
    #    batch_size = self.Y.shape[0]
    #    self.cost = -np.sum(np.log(self.final_activation[np.arange(batch_size), self.Y])) / batch_size

    #def __sparse_categorical_cross_entropy_gradient(self):
    #    batch_size = self.Y.shape[0]
    #    grad = np.zeros_like(self.final_activation)
    #    grad[np.arange(batch_size), self.Y] = -1 / self.final_activation[np.arange(batch_size), self.Y]
    #    return grad / batch_size

    def __mse_cost(self):
        self.cost = np.sum((self.final_activation - self.Y) ** 2) / self.m

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
        linear = lambda x : x
        self.final_activation, finalC = self.__forward_prop_calcs(A, Wn, bn, linear)
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

    def __gradient_descent(self):
        final_derivative = 2 * (self.final_activation - self.Y)    # This is dA4
        linear_deriv = lambda x : 1
        lcache, acache = self.caches[-1]
        dZ = final_derivative * linear_deriv(acache)                # We use it to calculate dZ4
        A, W, b = lcache                                            # From our cache, we get W4, b4, and A3
        m = A.shape[1]                                              # We see how many columns A3 had for calculating
        dW = np.dot(dZ, (A.T)) / m                                  # We calculate dW4
        db = np.sum(dZ, axis=1, keepdims=True) / m                  # We calculate db4
        dA = np.dot(W.T, dZ)                                        # BACKPROP STEP - we calculate dA3
        self.gradients["dW" + str(self.n-1)] = dW                   # We add dW4 and db4 to scale our parameters later.
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
            self.__mse_cost()
            self.__gradient_descent()
            print(f"Epoch {epoch + 1}: Cost = {self.cost}")
            print(f"Final Activations: {self.final_activation}\n\n")