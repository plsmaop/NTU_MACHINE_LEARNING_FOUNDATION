#-*- coding: utf-8 -*-
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
from pla import PLA

class Q6(PLA):
    def __init__(self, file_path, input_dimension):
        super(Q6, self).__init__(file_path, input_dimension)
        self.__freqs = []

    def __update_w_and_freqs_fn(self, round):
        def update_w_and_freqs(w, x, y):
            self.__freqs[round] += 1
            return w + y * x

        return update_w_and_freqs

    def __run(self, repeated_times):
        """
            Q6
        """

        self.__freqs = np.zeros(repeated_times)

        for i in range(repeated_times):
            super(Q6, self).run_with_random_cycle(self.__update_w_and_freqs_fn(i))

        return self.__freqs

    def run_and_show_histogram(self, repeated_times):
        h = self.__run(repeated_times)
        print('Q6: average number of updates:', reduce(lambda x, y: x+y, h) / repeated_times)
        plt.hist(h)
        plt.show()
