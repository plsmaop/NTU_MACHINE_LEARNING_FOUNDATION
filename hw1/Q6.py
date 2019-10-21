#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from pla import PLA

class Q6(PLA):
    def __init__(self, file_path, input_dimension):
        super(Q6, self).__init__(file_path, input_dimension)
        self.freqs = []

    def update_freqs_fn(self, round):
        def update_freqs():
            self.freqs[round] += 1

        return update_freqs

    def __run(self, repeated_times):
        """
            Q6
        """

        self.freqs = np.zeros(repeated_times)

        for i in range(repeated_times):
            super(Q6, self).run_with_random_cycle(1, self.update_freqs_fn(i))

        return self.freqs

    def run_and_show_histogram(self, repeated_times):
        h = self.__run(repeated_times)
        plt.hist(h)
        plt.show()
