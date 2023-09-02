from Nets.model_utils import NeuralNetwork
from Nets.activation_utils import *


class SoftMaxNN(NeuralNetwork):
    def __init__(self, n_sizes, lr, train_data, epochs=None, gs=None, gsp=None):

        n = len(n_sizes)
        if gs is None:
            gs = [relu for _ in range(n - 1)]
            gs.append(softmax)
            gsp = [drelu for _ in range(n - 1)]
            gsp.append(dsoftmax)
        if epochs is None:
            epochs = 1000

        assert gs[-1] == softmax and gsp[-1] == dsoftmax and train_data[1].shape[0] == n_sizes[-1]

        NeuralNetwork.__init__(self, n_sizes, lr, train_data, epochs, gs, gsp)

    def __cost(self):  # Uses categorical cross entropy to compute cost.
        m, Y, AL = self.m, self.Y, self.final_activation
        cost = (-1 / m) * np.sum(Y * np.log(AL))
        self.cost = np.squeeze(cost)

    def fdZ_calc(self):
        return self.final_activation - self.Y
