# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


class DecisionStump():
    def __init__(self, data_size, experiment_times, noise_prob):
        self.__data_size = data_size
        self.__experiment_times = experiment_times
        self.__noise_prob = noise_prob

    @staticmethod
    def __sign(number):
        return 1 if number >= 0 else -1

    def __gen_data(self):
        # step (a)
        X = np.random.uniform(-1, 1, size=(self.__data_size))

        # step(b)
        Y = map(lambda x: x if np.random.uniform() > self.__noise_prob else -1*x,
                [self.sign(x) for x in X])

        return X, Y

    def __run(self): pass
