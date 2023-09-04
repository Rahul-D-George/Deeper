from Datasets.training_set_utils import training_set_create
from Nets.model_utils import NeuralNetwork
from Nets.multiclass_utils import SoftMaxNN
import numpy as np


def binary_class():
    train_x, train_y = training_set_create()
    genshinNet = NeuralNetwork([np.shape(train_x)[0], 120, 60, 30, 15, 7, 1], 0.0075, [train_x, train_y], epochs=1000)
    genshinNet.train()
    genshinNet.predict(train_x, train_y)


def multi_class():
    train_x, train_y = training_set_create(type="age")
    genshinNet = SoftMaxNN([np.shape(train_x)[0], 120, 60, 33], 0.0075, [train_x, train_y], epochs=500)
    genshinNet.train()


multi_class()
