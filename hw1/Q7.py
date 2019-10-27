#-*- coding: utf-8 -*-
import math
import copy
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
from pla import PLA


class Q7(PLA):
    def __init__(self, file_path, verify_set_path, input_dimension):
        super(Q7, self).__init__(file_path, input_dimension)
        self.__initial_error_rate = math.inf
        self.__current_error_rate = math.inf
        self.__total_error_rate = 0
        self.__current_update_times = 0
        self.__best_w = np.zeros(self.input_dimension + 1)

        self.__test_X, self.__test_Y = super(Q7, self).load_data_from_file(verify_set_path, input_dimension)

    @staticmethod
    def __error_rate(X, Y, w):
        return reduce(lambda x, y: x+y, map(lambda x, y: 1.0 if super(Q7, Q7).sign(np.dot(x.T, w)) != y else 0, X, Y)) / len(X)

    def verify(self):
        error_rate = self.__error_rate(self.__test_X, self.__test_Y, self.__best_w)
        self.__total_error_rate += error_rate
        return error_rate

    def __run(self, repeated_times, max_update_times):

        def is_exceed_max_update():
            return self.__current_update_times > max_update_times

        # pocket algorithm
        def update_w(w, x, y):
            # update
            self.__current_update_times += 1
            new_w = w + y * x
            error_rate = self.__error_rate(self.X, self.Y, new_w)

            # pocket
            if self.__current_error_rate > error_rate:
                self.__current_error_rate = error_rate
                self.__best_w = copy.copy(new_w)

            return new_w

        # initialize error_rate
        # initial self.__best_w is 0
        self.__initial_error_rate = self.__error_rate(self.X, self.Y, self.__best_w)
        self.__current_error_rate = copy.copy(self.__initial_error_rate)
        
        for i in range(repeated_times):
            super(Q7, self).run_with_random_cycle(is_exceed_max_update, update_w)
            print('Round', i, ': Error Rate on trainning data is', self.__current_error_rate, ', on verify data is', self.verify())
            self.__current_update_times = 0
            self.__current_error_rate = copy.copy(self.__initial_error_rate)
            self.__best_w = np.zeros(self.input_dimension + 1)


        return self.__total_error_rate / repeated_times

    def run_and_show_histogram(self, repeated_times, max_update_times):
        r = self.__run(repeated_times, max_update_times)
        print('Q7: average error rate on verification set:', r)
        # plt.hist(h)
        # plt.show()