from numpy.random import uniform as u
from Nets.activation_utils import *


class NeuralNetwork:

    def __init__(self, n_sizes, lr, train_data, epochs=None, gs=None, gsp=None):

        self.n = len(n_sizes)
        self.params = {}  # Uses random initialisation.
        for l in range(1, self.n):
            self.params['W' + str(l)] = np.random.randn(n_sizes[l], n_sizes[l - 1]) / np.sqrt(
                n_sizes[l - 1])
            self.params['b' + str(l)] = np.zeros((n_sizes[l], 1))

        self.lr = lr
        self.X = np.array(train_data[0])
        self.Y = np.array(train_data[1])
        self.caches = [[]]
        self.m = self.Y.shape[1]

        self.final_activation = np.empty(np.shape(self.Y))
        self.acc = 0
        self.gradients = {}
        self.cost = -1

        if gs is None:  # Activations by default are Relu's and sigmoid in the final layer.
            self.gs = [relu for _ in range(self.n - 1)]
            self.gs.append(sigmoid)
            self.gsp = [drelu for _ in range(self.n - 1)]
            self.gsp.append(dsigmoid)
        else:
            self.gs = gs
            self.gsp = gsp

        if epochs is None:  # Epochs by default are 100.
            self.epochs = 100
        else:
            self.epochs = epochs

    def __cost(self):  # Uses cross entropy to compute cost.
        m, Y, AL = self.m, self.Y, self.final_activation
        cost = (-1 / m) * np.sum(Y * np.log(AL + 1e-8) + (1 - Y) * np.log(1 - AL + 1e-8))
        self.cost = np.squeeze(cost)

    @staticmethod
    def __forward_calc(Ap, W, b, actv):
        Z = np.dot(W, Ap) + b
        return actv(Z), ((Ap, W, b), Z)

    def __forward_prop(self):
        caches = []
        A = self.X
        for l in range(1, self.n - 1):
            W = self.params["W" + str(l)]
            b = self.params["b" + str(l)]
            A_prev = A.copy()
            Z = np.dot(W, A_prev) + b
            cache = ((A_prev, W, b), Z)
            A = self.gs[l](Z)
            caches.append(cache)
        Wn = self.params["W" + str(self.n - 1)]
        bn = self.params["b" + str(self.n - 1)]
        self.final_activation, finalC = self.__forward_calc(A, Wn, bn, self.gs[-1])
        caches.append(finalC)
        self.caches = caches

    def final_deriv(self):
        return - (np.divide(self.Y, self.final_activation)
                                         - np.divide(1 - self.Y, 1 - self.final_activation))

    def __gradient_descent(self):
        final_activation_derivative = self.final_deriv()
        lcache, acache = self.caches[-1]
        dZ = final_activation_derivative * self.gsp[-1](acache)     #
        A, W, b = lcache
        m = A.shape[1]
        dW = np.dot(dZ, A.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA = np.dot(W.T, dZ)
        self.gradients["dW" + str(self.n-1)] = dW
        self.gradients["db" + str(self.n-1)] = db

        for l in range(len(self.caches)-2, -1, -1):
            lcache, acache = self.caches[l]
            dZ = dA * self.gsp[l](acache)
            A, W, b = lcache
            m = A.shape[1]
            dW = np.dot(dZ, A.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dA = np.dot(W.T, dZ)
            self.gradients["dW" + str(l+1)] = dW
            self.gradients["db" + str(l+1)] = db

        for l in range(self.n-1):
            self.params["W" + str(l+1)] = self.params["W" + str(l+1)] - self.gradients["dW" + str(l+1)] * self.lr
            self.params["b" + str(l+1)] = self.params["b" + str(l+1)] - self.gradients["db" + str(l+1)] * self.lr

    def train(self):
        for epoch in range(0, self.epochs):
            self.__forward_prop()
            self.__cost()
            self.__gradient_descent()
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Cost = {self.cost}")

    def predict(self, test_x, test_y):
        self.X = test_x
        self.__forward_prop()
        probs = np.where(self.final_activation > 0.5, 1, 0)
        acc = np.sum(probs == test_y) / test_y.shape
        print(f"Percentage of accurate guesses: {100*acc[1]}%")
