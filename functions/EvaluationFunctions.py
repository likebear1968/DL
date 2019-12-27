import numpy as np
import itertools
import inspect
from enum import Enum
from functions.LossFunctions import categorical_cross_entropy

class TYPE(Enum): R2=1; MSE=2; MAE=3; ACC=4; PRC=5; RCL=6; F1=7; MTR=8; PPX=9

class Metrics():
    def __init__(self, mtype, ncls=2):
        self.ncls = ncls
        if mtype == TYPE.R2:  self.func = r2
        if mtype == TYPE.MSE: self.func = mse
        if mtype == TYPE.MAE: self.func = mae
        if mtype == TYPE.ACC: self.func = accuracy
        if mtype == TYPE.PRC: self.func = precision
        if mtype == TYPE.RCL: self.func = recall
        if mtype == TYPE.F1:  self.func = f1
        if mtype == TYPE.MTR: self.func = confusion_matrix
        if mtype == TYPE.PPX: self.func = perplexity
    
    def metrics(self, obs, prd):
        if 'cls' in inspect.signature(self.func).parameters:
            return self.func(obs, prd, list(range(self.ncls)))
        return self.func(obs, prd)

def r2(obs, prd):
    t = np.array(obs)
    y = np.array(prd)
    denom = np.sum((t - np.mean(t)) ** 2)
    if denom > 0:
        return 1 - np.sum((t - y) ** 2) / denom
    return 1 - np.sum((t - y) ** 2)

def mse(obs, prd):
    t = np.array(obs)
    y = np.array(prd)
    return np.average((t - y) ** 2, axis=0)

def mae(obs, prd):
    t = np.array(obs)
    y = np.array(prd)
    return np.average(np.abs(y - t), axis=0)

def confusion_matrix(obs, prd, cls=[0,1]):
    s = len(cls)
    t = np.array(obs)
    y = np.array(prd)
    matrix = []
    for a, b in itertools.product(cls, repeat=2):
        matrix.append(len(np.where((t == a) & (y == b))[0]))
    return np.reshape(matrix, (s, s))

def accuracy(obs, prd, cls=[0,1], mtr=None):
    if mtr is None:
        mtr = confusion_matrix(obs, prd, cls)
    return np.sum(np.diag(mtr)) / np.sum(mtr)

def precision(obs, prd, cls=[0,1], mtr=None, delta=1e-7):
    if mtr is None:
        mtr = confusion_matrix(obs, prd, cls)
    return np.average((np.diag(mtr) + delta) / (np.sum(mtr, axis=0) + delta))

def recall(obs, prd, cls=[0,1], mtr=None, delta=1e-7):
    if mtr is None:
        mtr = confusion_matrix(obs, prd, cls)
    return np.average((np.diag(mtr) + delta) / (np.sum(mtr, axis=1) + delta))

def f1(obs, prd, cls=[0,1], mtr=None, delta=1e-7):
    if mtr is None:
        mtr = confusion_matrix(obs, prd, cls)
    r = recall(None, None, None, mtr, delta)
    p = precision(None, None, None, mtr, delta)
    return (2 * r * p) / (r + p)

def perplexity(obs, prd, delta=1e-7):
    l, _, _, _ = categorical_cross_entropy(obs, prd, delta)
    return 2 ** l
