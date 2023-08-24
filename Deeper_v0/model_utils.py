import numpy as np
from numpy.random import uniform as u
from math import sqrt
from _collections import defaultdict

class NeuralNetwork:

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    # lr        =   Learning Rate
    # n_sizes   =   Size of each layer in neurons, passed as a list. Includes the
    #               number of inputs. E.g.: [3, 4, 3, 2, 1]
    # w, b      =   Arrays of parameters for each layer, i.e: b[3] corresponds to b for layer 3.
    # cache     =   Used to store Z[l] and A[l] in a (n, 2) array. DOES NOT INCLUDE A0.
    def __init__(self, n_sizes, lr, train_data, g=None, gprime=None, epochs=None):
        assert n_sizes[0] == len(train_data[0])
        assert n_sizes[-1] == 1
        self.n = len(n_sizes)

        # Xavier/Glorot Initialisation
        self.params = {}
        #self.W = []
        for l in range(1, self.n):
            xgb = sqrt(6) / sqrt(n_sizes[l] + n_sizes[l - 1])
            self.params["W"+str(l)] = np.array([u(-xgb, xgb, n_sizes[l - 1]) for _ in range(n_sizes[l])])
            self.params["b"+str(l)] = np.zeros((n_sizes[l], 1))
            #self.W.append(np.array([u(-xgb, xgb, n_sizes[l - 1]) for _ in range(n_sizes[l])]))

        #self.b = [np.array([[0] for _ in range(n_sizes[i])]) for i in range(1, self.n)]
        self.lr = lr
        self.X = np.array(train_data[0])
        self.Y = np.array(train_data[1])
        self.cache = [[np.array([]), np.array([])] for _ in range(self.n)]
        self.cache[0][1] = self.X
        self.m = len(self.Y)
        self.cost = 0
        if g is None:
            self.g = [np.tanh for _ in range(self.n - 1)]
            self.g.append(lambda x: x)
            self.gprime = [self.tanh_derivative for _ in range(self.n - 1)]
            self.gprime.append(lambda x: 1)
        else:
            self.g = g
            self.gprime = gprime
        if epochs is None:
            self.epochs = 100
        else:
            self.epochs = epochs

    def __mse_cost(self):
        self.cost = np.sum((self.cache[-1][1] - self.Y) ** 2) / self.m

    def __forward_prop(self):
        A = self.X
        for l in range(1, self.n):
            W = self.params["W" + str(l)]
            b = self.params["b" + str(l)]
            Z = np.dot(W, A) + b
            A = self.g[l](Z)
            self.cache[l] = [Z, A]

    def __gradient_descent(self):
        dA = 2 * (self.cache[-1][1] - self.Y)
        for l in range(self.n - 1, 0, -1):
            dZ = dA * self.gprime[l](self.cache[l][0])
            dW = (1 / self.m) * np.dot(dZ, self.cache[l - 1][1].T)
            dB = (1 / self.m) * np.sum(dZ, axis=1, keepdims=True)
            W = self.params["W"+str(l)]
            b = self.params["b"+str(l)]
            self.params["W"+str(l)] = W - (self.lr * dW)
            self.params["b"+str(l)] = b - (self.lr * dB)
            dA = np.dot(W.T, dZ)

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

    def fprop_z_calc(self, A, W, b):
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

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
        cost = np.squeeze(cost)
        return cost

