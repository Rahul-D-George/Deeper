from training_set_utils import training_set_create
from model_utils import NeuralNetwork

data = training_set_create()

GenshinNet = NeuralNetwork([len(data[0]), 480, 240, 120, 60, 30], 0.01, data, epochs=30)
GenshinNet.train()

#GenshinNet.predict()