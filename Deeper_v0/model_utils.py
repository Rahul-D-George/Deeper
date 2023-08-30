from numpy.random import uniform as u
from activation_utils import *

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(Y * np.log(AL+1e-8) + (1 - Y) * np.log(1 - AL+1e-8))
    cost = np.squeeze(cost)
    return cost

class NeuralNetwork:

    def __init__(self, n_sizes, lr, train_data, epochs=None, gs=None, gsp=None):

        self.n = len(n_sizes)

        self.params = {}  # Uses random initialisation.
        for l in range(1, self.n):
            self.params['W' + str(l)] = np.random.randn(n_sizes[l], n_sizes[l - 1]) * 0.01
            self.params['b' + str(l)] = np.zeros((n_sizes[l], 1))

        # Xavier/Glo rot Initialisation - unimplemented for now.
        # self.params = {}
        # for l in range(1, self.n):
        #     xgb = sqrt(6) / sqrt(n_sizes[l] + n_sizes[l - 1])
        #     self.params["W"+str(l)] = np.array([u(-xgb, xgb, n_sizes[l - 1]) for _ in range(n_sizes[l])])
        #     self.params["b"+str(l)] = np.zeros((n_sizes[l], 1))

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

    @staticmethod  # Returns A, and a cache of the form (linear cache, activation cache).
    def __forward_calc(Ap, W, b, actv):
        Z = np.dot(W, Ap) + b
        return actv(Z), ((Ap, W, b), Z)

    def __forward_prop(self):
        caches = []
        A = self.X
        for l in range(1, self.n - 1):
            W = self.params["W" + str(l)]
            b = self.params["b" + str(l)]
            A_prev = A
            A, cache = self.__forward_calc(A_prev, W, b, self.gs[l])
            caches.append(cache)
        Wn = self.params["W" + str(self.n - 1)]
        bn = self.params["b" + str(self.n - 1)]
        self.final_activation, finalC = self.__forward_calc(A, Wn, bn, self.gs[-1])
        caches.append(finalC)
        self.caches = caches

    def __gradient_descent(self):
        final_activation_derivative = - (np.divide(self.Y, self.final_activation)
                                         - np.divide(1 - self.Y, 1 - self.final_activation))

        lcache, acache = self.caches[-1]

        dZ = final_activation_derivative * self.gsp[-1](acache)     #
        A, W, b = lcache
        m = A.shape[1]
        dW = np.dot(dZ, (A.T)) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA = np.dot(W.T, dZ)
        self.gradients["dW" + str(self.n-1)] = dW
        self.gradients["db" + str(self.n-1)] = db

        for l in range(len(self.caches)-2, -1, -1):                 # We want to start with our first unaccessed cache.
            lcache, acache = self.caches[l]                         # We extract the separate caches
            dZ = dA * self.gsp[l](acache)                           # We use dA3 and Z3 to calculate dZ3
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
        for epoch in range(0, self.epochs):
            self.__forward_prop()
            self.__cost()
            self.__gradient_descent()
            if (epoch % 100 == 0):
                print(f"Epoch {epoch}: Cost = {self.cost}")

    def predict(self, test_x, test_y):
        self.X = test_x
        self.__forward_prop()
        probs = np.where(self.final_activation > 0.5, 1, 0)
        acc = np.sum(probs == test_y) / test_y.shape
        print(f"Percentage of accurate guesses: {acc[1]}")
