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
        assert n_sizes[-1] == 1
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

    def __mse_cost(self):
        self.cost = np.sum((self.final_activation - self.Y) ** 2) / self.m

    @staticmethod
    def __forward_prop_calcs(A_prev, W, b, activation):
        Z = np.dot(W, A_prev) + b
        return activation(Z), ((A_prev, W, b), Z) # Returns A and Cache

    def __forward_prop(self):
        caches = []
        A = self.X
        for l in range(1, self.n):
            W = self.params["W" + str(l)]
            b = self.params["b" + str(l)]
            A_prev = A
            tanh = lambda x: np.tanh(x)
            A, cache = self.__forward_prop_calcs(A_prev, W, b, tanh)
            caches.append(cache)
        Wn = self.params["W" + str(self.n)]
        bn = self.params["b" + str(self.n)]
        linear = lambda x : np.identity(x)
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
        final_derivative = -2 * (self.final_activation - self.Y)
        linear_deriv = lambda x : 1
        self.__backward_prop_calcs(-1, self.n, linear_deriv, final_derivative)

        tanh_deriv = lambda x : 1 - np.tanh(x)**2
        for l in range(self.n - 2, -1, -1):
            self.__backward_prop_calcs(l, l+1, tanh_deriv, self.gradients["dA"+str(l+1)])


    def train(self):
        for epoch in range(self.epochs):
            self.__forward_prop()
            self.__mse_cost()
            self.__gradient_descent()
            print(f"Epoch {epoch + 1}: Cost = {self.cost}")

    def predict(self):
        rate = 0
        for i in range(len(self.Y)):
            if self.cache[-1][1][i] == self.Y[i]:
                rate += 1
        print(f"Accuracy on training set: {rate / len(self.Y)}")


class NeuralNetworkStructured:
    def __int__(self):
        self.exists = True
        self.parameters = {}

    # Function to randomly initialise L-layer DNN weights and biases.

    def initialize_parameters_deep(self, layer_dims):
        np.random.seed(3)
        L = len(layer_dims)
        for l in range(1, L):
            self.parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
            self.parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
            assert (self.parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
            assert (self.parameters['b' + str(l)].shape == (layer_dims[l], 1))

    @staticmethod
    def fprop_z_calc(A, W, b):
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        return Z, cache

    def fprop_a_calc(self, A_prev, W, b, activation):
        Z, linearcache = self.fprop_z_calc(A_prev, W, b)
        A, activationcache = activation(Z), Z
        cache = (linearcache, activationcache)
        return A, cache

    def L_model_forward(self, X, parameters):
        caches = []
        A = X
        L = len(parameters) // 2
        for l in range(1, L):
            W = parameters["W" + str(l)]
            b = parameters["b" + str(l)]
            A_prev = A
            tanh = lambda x: np.tanh(x)
            A, cache = self.fprop_a_calc(self, A_prev, W, b, tanh)
            caches.append(cache)
        linear = lambda x : x
        AL, cache = self.fprop_a_calc(A, parameters["W" + str(L)], parameters["b" + str(L)], linear)
        caches.append(cache)
        return AL, caches

    @staticmethod
    def compute_cost(AL, Y):
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
        cost = np.squeeze(cost)
        return cost

    @staticmethod
    def linear_backward(dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = np.dot(dZ, (A_prev.T)) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    @staticmethod
    def linear_activation_backward(self, dA, cache, backtivation):
        linear_cache, activation_cache = cache
        dZ = backtivation(dA, activation_cache)
        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    @staticmethod
    def L_model_backward(self, AL, Y, caches):

        grads = {}
        L = len(caches)  # the number of layers
        Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        current_cache = caches[L - 1]
        dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dAL, current_cache, "sigmoid")
        grads["dA" + str(L - 1)] = dA_prev_temp
        grads["dW" + str(L)] = dW_temp
        grads["db" + str(L)] = db_temp

        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dA_prev_temp, current_cache, "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
        return grads