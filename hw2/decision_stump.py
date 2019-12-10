# -*- coding: utf-8 -*-
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt


# Eout = 0.5 + 0.3*s(abs(theta) - 1)
NOISE_PROB = 0.2
CONST = 0.5
SCALAR = 0.3


class EOUT():
    def __init__(self, const, scalar):
        self.__const = const
        self.__scalar = scalar

    def calculate(self, s, theta) -> float:
        """
            Calculate Eout by s and theta
            input:
                s: 1 or -1
                theta: float between -1 and 1
        """

        return self.__const + self.__scalar * s * (abs(theta) - 1)


class DecisionStump():
    def __init__(self, data_size, experiment_times):
        self.__data_size = data_size
        self.__experiment_times = experiment_times
        self.__eout_calculator = EOUT(CONST, SCALAR)

    @staticmethod
    def __gen_theta(X) -> [float]:
        # edge case and interval
        return [(-1 + X[0])/2, (1 + X[-1])/2] + [(X[ind] + X[ind+1])/2 for ind in range(len(X) - 1)]

    @staticmethod
    def __sign(number) -> int:
        return 1 if number >= 0 else -1

    def __gen_data(self) -> ([float], [int]):
        # step (a)
        X = np.random.uniform(-1, 1, size=(self.__data_size))

        # step(b)
        Y = list(map(lambda x: x if np.random.uniform() > NOISE_PROB else -1*x,
                     [self.__sign(x) for x in X]))

        return X, Y

    def __run(self, X, Y) -> (int, float, float):
        S = -1
        T = 0
        E = len(X)

        # sort
        sorted_X_with_Y = [(x, y) for x, y in zip(X, Y)]
        sorted_X_with_Y.sort(key=lambda item: item[0])

        theta = self.__gen_theta([xy[0] for xy in sorted_X_with_Y])

        for s in [-1, 1]:
            for t in theta:
                def h(x): return s*self.__sign(x-t)

                # calcute
                err = reduce(
                    lambda x, y: x+y, map(lambda xy: 1 if h(xy[0]) != xy[1] else 0, sorted_X_with_Y))

                if E > err:
                    S = s
                    T = t
                    E = err

        return S, T, E/len(X)

    def run_and_render_histogram(self):
        """
            run the decision stump algorithm and render the histogram
        """

        Ein = []
        Eout = []

        for _ in range(self.__experiment_times):
            X, Y = self.__gen_data()
            s, t, ein = self.__run(X, Y)
            eout = self.__eout_calculator.calculate(s, t)
            Ein.append(ein)
            Eout.append(eout)

        print('Ein:', np.mean(Ein), ',Eout:', np.mean(Eout))
        plt.hist(list(map(lambda ein, eout: ein - eout, Ein, Eout)))
        plt.xlabel('Ein - Eout')
        plt.title('Histogram')
        plt.show()
