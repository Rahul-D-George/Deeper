from training_set_utils import training_set_create
from model_utils import NeuralNetwork
import numpy as np

train_x, train_y = training_set_create()

genshinNet = NeuralNetwork([np.shape(train_x)[0], 120, 60, 30, 10, 1], 0.0075, [train_x, train_y], epochs=500)
genshinNet.train()
genshinNet.predict(train_x, train_y)
