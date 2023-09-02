from model_utils import NeuralNetwork
from activation_utils import *


class SoftMaxNN(NeuralNetwork):
    def __init__(self, n_sizes, lr, train_data, epochs=None, gs=None, gsp=None):

        if gs is None:
            self.gs = [relu for _ in range(self.n - 1)]
            self.gs.append(softmax)
            self.gsp = [drelu for _ in range(self.n - 1)]
            self.gsp.append(dsoftmax)
        if epochs is None:
            epochs = 1000

        assert gs[-1] == softmax and gsp[-1] == dsoftmax and train_data[1][0] == n_sizes[-1]

        NeuralNetwork.__init__(self, n_sizes, lr, train_data, epochs, gs, gsp)

    def final_deriv(self):
        return self.final_activation - self.Y
