import numpy as np


def tanh(x):
    return np.tanh(x)


def dtanh(x):
    return 1 - (tanh(x)**2)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def dsigmoid(x):
    sig = sigmoid(x)
    return sig*(1-sig)


def relu(x):
    return np.maximum(0, x)


def drelu(x):
    return np.where(x > 0, 1, 0)


def softmax(x):
    exp = np.exp(x)
    return exp/np.sum(exp)


def dsoftmax(x):
    print(x.shape)
    s = x.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)