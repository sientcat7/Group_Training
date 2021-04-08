# -*- codeing = utf-8 -*-
# @Time : 2021/4/8 12:17
# @Author : 陈铿任
# @File ： classdef.py
# @Software： PyCharm

import numpy as np
from math import sqrt

class LinearRegression:
    def __init__(self):
        self.vab_ = None
        self.interception = None
        self._w = None

    def fit(self,X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        X_b = np.hstack([np.ones(len(X_train), 1), X_train])
        self.w = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.interception_ = self._w[0]
        self.vab_ = self._theta[1:]
        return self

    def predict(self, X_predict):
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones(len(X_predict), 1), X_predict])
        return X_b.dot(self._w)
    def _repr(self):
        return LinearRegression()


def mean_squared_error(y_true, predict_y):
    assert len(y_true) == len(predict_y), \
        "the size of y_true must be equal to the size of y_predict"
    return np.sum((y_true - predict_y) ** 2) / len(y_true)

def root_mean_squared_error(y_true, predict_y):
    return sqrt(mean_squared_error(y_true, predict_y))

def mean_absoluted_error(y_true, predict_y):
    assert len(y_true) == len(predict_y), \
        "the size of y_true must be equal to the size of y_predict"
    return np.sum(np.absolute(y_true, predict_y)) / len(predict_y)

def re_score(y_true, predict_y):
    return 1-mean_squared_error(y_true, predict_y)/np.var(y_true)
