# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


class LogisticRegression():
    def __init__(self: object, eta: float, turns: int, _GD: 'function'):
        self.__eta = eta
        self.__turns = turns
        self.__GD = _GD

    @staticmethod
    def open_file(file_path: str) -> ([[float]], [float]):
        X, Y = [], []
        with open(file_path) as f:
            def append(line: [str]):
                line = list(map(lambda x: float(x), line.split(' ')[1:]))
                X.append(line[:-1])
                Y.append(line[-1])

            # force execute
            list(map(append, f))

        return X, Y

    def __caculate(self: object, X: [[float]], Y: [float]):
        print(X)
        print(Y)

    def caculate_and_plot(self: object, data_source: str):
        """
            return logistic regression result of the data_source
        """
        X, Y = self.open_file(data_source)
        return self.__caculate(X, Y)
