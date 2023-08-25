from training_set_utils import training_set_create
from model_utils import NeuralNetwork

data = training_set_create()

GenshinNet = NeuralNetwork([len(data[0]), 2000, 1000, 500, 200, 100, 50], 0.1, data, epochs=30)
GenshinNet.train()

#GenshinNet.predict()