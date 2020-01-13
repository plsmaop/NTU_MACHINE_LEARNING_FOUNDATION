# -*- coding: utf-8 -*-
import numpy as np


class GradientDescent():
    def caculate(self: object, w: [float], X: [[float]], Y: [float]) -> float:
        g = np.zeros(len(X[0]))
        for n in range(len(X)):
            g += (self.sigmoid(-X[n].dot(w) * Y[n]) * (-Y[n]) * X[n])

        return g / len(X)

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))


class StocasticGradientDescent(GradientDescent):
    def __init__(self: object):
        self.__turn = 0

    def caculate(self: object, w: float, X: [[float]], Y: [float]) -> float:
        turn = self.__turn % len(X)

        self.__turn += 1
        return super().caculate(w, X[turn: turn + 1], Y[turn: turn + 1])
