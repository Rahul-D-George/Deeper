from training_set_utils import training_set_create
from model_utils import NeuralNetwork

data = training_set_create()

GenshinNet = NeuralNetwork([len(data[0]), 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], 0.02, data, epochs=50)
GenshinNet.train()