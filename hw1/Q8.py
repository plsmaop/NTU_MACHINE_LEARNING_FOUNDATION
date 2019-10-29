#-*- coding: utf-8 -*-
from functools import reduce
import copy
import matplotlib.pyplot as plt
from Q7 import Q7

class Q8(Q7):
    def __init__(self, file_path, verify_set_path, input_dimension):
        super(Q8, self).__init__(file_path, verify_set_path, input_dimension)

    def __update_w(self, w, x, y):
        # update
        self._Q7__current_update_times += 1
        new_w = w + y * x

        self._Q7__current_error_rate = self._Q7__error_rate(self.X, self.Y, new_w)
        self._Q7__best_w = copy.copy(new_w)

        # print(new_w)
        return new_w

    def run_and_show_histogram(self, repeated_times, max_update_times):
        f = super(Q8, self).run(repeated_times, max_update_times, self.__update_w)
        print('Q8: average error rate on the test set:', reduce(lambda x, y: x+y, f) / repeated_times)
        super(Q8, self).show_histogram(f)
