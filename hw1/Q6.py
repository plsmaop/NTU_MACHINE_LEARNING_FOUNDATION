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

        def is_exceed_max_update(): return False

        self.__freqs = np.zeros(repeated_times)

        for i in range(repeated_times):
            super(Q6, self).run_with_random_cycle(is_exceed_max_update, self.__update_w_and_freqs_fn(i))
            print('Round', i, ': update', self.__freqs[i], 'times')

        return self.__freqs

    def run_and_show_histogram(self, repeated_times):
        f = self.__run(repeated_times)
        print('Q6: average number of updates:', reduce(lambda x, y: x+y, f) / repeated_times)
        plt.hist(f, bins=[int(i) for i in range(int(np.amin(f)), int(np.amax(f)+1))])
        plt.title('Number of Updates Versus Frequency')
        plt.xlabel('Number of Updates')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
