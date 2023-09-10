import numpy as np
import h5py
from Nets.model_utils import NeuralNetwork

train_dataset = h5py.File('train_catvnoncat.h5', "r")
unmod_train_x = np.array(train_dataset["train_set_x"][:])
unmod_train_y = np.array(train_dataset["train_set_y"][:])

test_dataset = h5py.File('test_catvnoncat.h5', "r")
unmod_test_x = np.array(test_dataset["test_set_x"][:])
unmod_test_y = np.array(test_dataset["test_set_y"][:])

train_y = unmod_train_y.reshape((1, unmod_train_y.shape[0]))
test_y = unmod_test_y.reshape((1, unmod_test_y.shape[0]))
train_x_flatten = unmod_train_x.reshape(unmod_train_x.shape[0], -1).T
test_x_flatten = unmod_test_x.reshape(unmod_test_x.shape[0], -1).T

train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

catnn = NeuralNetwork([np.shape(train_x)[0], 4, 4, 1], 0.0075, [train_x, train_y], epochs=1200)
catnn.train()
catnn.predict(test_x, test_y)
