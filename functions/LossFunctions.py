import numpy as np
from functions.UtilityFunctions import id_to_category
from functions.EvaluationFunctions import mse

def linear(obs, prd):
    y = np.reshape(prd, (-1, 1))
    t = np.reshape(obs, np.shape(y))
    l = np.average(mse(t, y))
    d = (y - t) / np.shape(t)[0]
    return l, d, t, y

def binary_cross_entropy(obs, prd, delta=1e-7):
    y = np.reshape(prd, (-1, 1))
    y = np.argmax(np.c_[1 - y, y], axis=1)
    t = np.reshape(obs, np.shape(y))
    y2 = np.reshape(prd, np.shape(y))
    l = -np.average(t * np.log(y2 + delta) + (1 - t) * np.log(1 - y2 + delta))
    d = np.reshape((np.reshape(prd, np.shape(y)) - t) / np.shape(t)[0], np.shape(prd))
    return l, d, t, y

def categorical_cross_entropy(obs, prd, delta=1e-7):
    t = id_to_category(obs, prd)
    y = np.reshape(np.zeros_like(prd, dtype=int), (-1, np.shape(prd)[-1]))
    y[np.arange(np.shape(y)[0]), np.argmax(prd, axis=-1)] = 1
    y = np.reshape(y, np.shape(t))
    y2 = np.reshape(prd, np.shape(y))
    l = -np.average(t * np.log(y2 + delta))
    d = np.copy(prd)
    d[np.arange(np.shape(t)[0]), np.argmax(t, axis=-1)] -= 1
    d /= np.shape(t)[0]
    return l, d, np.argmax(t, axis=-1), np.argmax(y, axis=-1)
