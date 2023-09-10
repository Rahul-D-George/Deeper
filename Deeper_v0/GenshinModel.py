from Datasets.training_set_utils import training_set_create
from Nets.model_utils import NeuralNetwork
from Nets.multiclass_utils import ConvClassifier
import numpy as np


def binary_class():
    train_x, train_y = training_set_create()
    genshinNet = NeuralNetwork([np.shape(train_x)[0], 120, 60, 30, 15, 7, 1], 0.0075, [train_x, train_y], epochs=1000)
    genshinNet.train()
    genshinNet.predict(train_x, train_y)


def multi_class():
    train_x, train_y, test_x, test_y, train_names, test_names = training_set_create(type="age")
    genshinNet = ConvClassifier(train_x, train_y, epochs=30)
    genshinNet.train()
    results = genshinNet.predict(train_x)
    for i in range(len(train_names)):
        print(f"Prediction for {train_names[i]}: {results[i]}. Actual Value: {train_y[i]}")


multi_class()
