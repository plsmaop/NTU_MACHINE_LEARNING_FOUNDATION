# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from gradient_descent import GradientDescent


class LogisticRegression():
    def __init__(self: object, eta: float, turns: int, GD: GradientDescent, SGD: GradientDescent):
        self.__eta = eta
        self.__turns = turns
        self.__GD = GD
        self.__SGD = SGD

    @staticmethod
    def open_file(file_path: str) -> ([[float]], [float]):
        X, Y = [], []
        with open(file_path) as f:
            def append(line: [str]):
                line = list(map(lambda x: float(x), line.split(' ')[1:]))
                X.append([0] + line[:len(line)-1])
                Y.append(line[-1])

            # force execute
            list(map(append, f))

        return np.array(X), np.array(Y)

    @staticmethod
    def sign(x: float) -> int:
        return 1 if x >= 0 else -1

    def __error(self: object, w: [float], X: [[float]], Y: [float]) -> float:
        return np.mean(list(map(lambda x, y: 1 if self.sign(x.dot(w)) != y else 0, X, Y)))

    def __caculate_and_plot(self: object, X: [[float]], Y: [float], test_X: [[float]], test_Y: [float]) -> ([[float]], [[float]]):
        w = np.zeros(len(X[0]))
        sgd_w = np.zeros(len(X[0]))

        Ein = np.zeros(self.__turns)
        Eout = np.zeros(self.__turns)
        sgd_Ein = np.zeros(self.__turns)
        sgd_Eout = np.zeros(self.__turns)
        for ind in range(self.__turns):
            g = self.__GD.caculate(w, X, Y)
            sgd_g = self.__SGD.caculate(w, X, Y)

            w = w - self.__eta * g
            sgd_w = sgd_w - self.__eta * sgd_g
            Ein[ind] = self.__error(w, X, Y)
            sgd_Ein[ind] = self.__error(sgd_w, X, Y)

            Eout[ind] = self.__error(w, test_X, test_Y)
            sgd_Eout[ind] = self.__error(sgd_w, test_X, test_Y)

        # plot
        plt.plot(range(len(Ein)), Ein, label='GD')
        plt.plot(range(len(sgd_Ein)), sgd_Ein, label='SGD')
        plt.title(f'eta = {self.__eta}')
        plt.ylabel('Ein')
        plt.xticks(np.arange(0, self.__turns - 1, 250))
        plt.legend()
        plt.show()

        # plot
        plt.plot(range(len(Eout)), Eout, label='GD')
        plt.plot(range(len(sgd_Eout)), sgd_Eout, label='SGD')
        plt.title(f'eta = {self.__eta}')
        plt.ylabel('Eout')
        plt.xticks(np.arange(0, self.__turns - 1, 250))
        plt.legend()
        plt.show()

    def caculate_and_plot(self: object, train_source: str, test_source: str):
        """
            return logistic regression result of the data_source
        """
        X, Y = self.open_file(train_source)
        test_X, test_Y = self.open_file(test_source)
        self.__caculate_and_plot(X, Y, test_X, test_Y)
