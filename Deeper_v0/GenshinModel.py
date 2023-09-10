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
    genshinNet = ConvClassifier(train_x, train_y, epochs=17)
    genshinNet.train()
    full_set = np.concatenate((train_x, test_x), axis=0)
    results = genshinNet.predict(full_set)
    ages = np.append(train_y, test_y)
    names = np.append(train_names, test_names)

    for i in range(len(np.append(train_names, test_names))):
        print(f"Prediction for {names[i]}: {results[i]}. Actual Value: {ages[i]}")

multi_class()
