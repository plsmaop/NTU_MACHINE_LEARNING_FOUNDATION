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
        self.__current_update_times = 0
        self.__best_w = np.zeros(self.input_dimension + 1)

        self.__test_X, self.__test_Y = super(Q7, self).load_data_from_file(verify_set_path, input_dimension)
        self.__freqs = []

    @staticmethod
    def error_rate(X, Y, w):
        return reduce(lambda x, y: x+y, map(lambda x, y: 1.0 if super(Q7, Q7).sign(np.dot(x.T, w)) != y else 0, X, Y)) / len(X)

    def __verify(self, round, X, Y, w):
        error_rate = self.error_rate(X, Y, w)
        self.__freqs[round] = error_rate
        return error_rate

    # pocket algorithm
    def __update_w(self, w, x, y):
        # update
        self.__current_update_times += 1
        new_w = w + y * x
        error_rate = self.error_rate(self.X, self.Y, new_w)

        # pocket
        if self.__current_error_rate > error_rate:
            self.__current_error_rate = error_rate
            self.__best_w = copy.copy(new_w)

        return new_w

    def run(self, repeated_times, max_update_times, update_w, verify):

        def is_exceed_max_update():
            return self.__current_update_times > max_update_times

        self.__freqs = np.zeros(repeated_times)

        # initialize error_rate
        # initial self.__best_w is 0
        self.__initial_error_rate = self.error_rate(self.X, self.Y, self.__best_w)
        self.__current_error_rate = copy.copy(self.__initial_error_rate)
        
        for i in range(repeated_times):
            super(Q7, self).run_with_random_cycle(is_exceed_max_update, update_w)
            print('Round', i, ': Error Rate on verification data is', verify(i, self.__test_X, self.__test_Y, self.__best_w), ', on training data is', self.__current_error_rate)
            self.__current_update_times = 0
            self.__current_error_rate = copy.copy(self.__initial_error_rate)
            self.__best_w = np.zeros(self.input_dimension + 1)

        return self.__freqs

    @staticmethod
    def show_histogram(f):
        plt.hist(f)
        plt.title('Error Rate Versus Frequency')
        plt.xlabel('Error Rate')
        plt.ylabel('Frequency')
        plt.show()

    def run_and_show_histogram(self, repeated_times, max_update_times):
        f = self.run(repeated_times, max_update_times, self.__update_w, self.__verify)
        print('Q7: average error rate on the test set:', reduce(lambda x, y: x+y, f) / repeated_times)
        self.show_histogram(f)
        
