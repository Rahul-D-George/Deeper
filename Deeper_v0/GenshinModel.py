from training_set_utils import training_set_create
from model_utils import NeuralNetwork

data = training_set_create()

GenshinNet = NeuralNetwork([len(data[0]), 4, 4, 4, 4, 1], 0.1, data, epochs=30)
GenshinNet.train()
