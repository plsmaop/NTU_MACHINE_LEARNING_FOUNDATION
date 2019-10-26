#-*- coding: utf-8 -*-
import math
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
from pla import PLA


class Q7(PLA):
    def __init__(self, file_path, verify_set_path, input_dimension):
        super(Q7, self).__init__(file_path, input_dimension)
        self.__current_w = np.zeros(self.input_dimension + 1)
        self.__initial_error_rate = math.inf
        self.__current_error_rate = math.inf
        self.__total_error_rate = 0
        self.__current_update_times = 0

        with open(verify_set_path, 'r') as file:
            pass

    @staticmethod
    def __error_rate(X, Y, w):
        return reduce(lambda x, y: x+y, map(lambda x, y: 1.0 if super(Q7, Q7).sign(np.dot(x.T, w)) != y else 0, X, Y)) / len(X)

    def __run(self, repeated_times, max_update_times):

        def is_exceed_max_update():
            self.__current_update_times += 1
            return self.__current_update_times > max_update_times

        # pocket algorithm
        def update_w(w, x, y):
            new_w = w + y * x
            error_rate = self.__error_rate(self.X, self.Y, new_w)
            self.__total_error_rate += error_rate
            if self.__current_error_rate > error_rate:
                self.__current_error_rate = error_rate
                self.__current_w = new_w

            print(new_w, self.__current_error_rate, self.__initial_error_rate, error_rate, self.__current_w, self.__current_update_times)
            return self.__current_w

        # initialize error_rate
        self.__initial_error_rate = self.__current_error_rate = self.__error_rate(self.X, self.Y, self.__current_w)
        
        for i in range(repeated_times):
            print(i)
            super(Q7, self).run_with_random_cycle(is_exceed_max_update, update_w)
            self.__current_update_times = 0
            self.__current_error_rate = self.__initial_error_rate


        return self.__total_error_rate / max_update_times

    def run_and_show_histogram(self, repeated_times, max_update_times):
        h = self.__run(repeated_times, max_update_times)
        print('Q7: average number of updates:', h / repeated_times)
        # plt.hist(h)
        # plt.show()