#-*- coding: utf-8 -*-
import random
import time
import copy
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
from pocket import Pocket


class Q7(Pocket):
    def __init__(self, file_path, verify_set_path, input_dimension):
        super(Q7, self).__init__(file_path, verify_set_path, input_dimension)
        self.__current_error_index = []
        self.__initial_error_index = []
        self.__current_w = np.zeros(input_dimension + 1)
        self.__current_update_times = 0

    @staticmethod
    def __error_rate(X, Y, w):
        return reduce(lambda x, y: x+y, map(lambda x, y: 1.0 if super(Q7, Q7).sign(np.dot(x.T, w)) != y else 0, X, Y)) / len(X)

    def __run(self, repeated_times, max_update_times):

        self.__current_update_times = 0
        def is_exceed_max_update():
            self.__current_update_times += 1
            return self.__current_update_times > max_update_times or len(self.__current_error_index) == 0

        # pocket algorithm
        def update_w(w, X, Y):
            rand_error_index = self.__current_error_index[random.randint(0, len(self.__current_error_index) - 1)]
            new_w = w + Y[rand_error_index] * X[rand_error_index]

            new_error_index = super(Q7, self).get_error_index(X, Y, new_w)
            if len(new_error_index) < len(self.__current_error_index):
                self.__current_error_index = copy.copy(new_error_index)
                self.__current_w = copy.copy(new_w)

            # print(self.Y[rand_error_index])
            # print(new_w, self.X[rand_error_index], self.Y[rand_error_index], self.__current_w, len(new_error_index), len(self.__current_error_index))
            return self.__current_w

        self.__current_error_index = self.__initial_error_index = super(Q7, self).get_error_index(self.X, self.Y, self.__current_w)

        for i in range(1):
            print(i)
            random.seed(time.time())
            super(Q7, self).train(is_exceed_max_update, update_w)
            self.__current_update_times = 0
            self.__current_error_index = copy.copy(self.__initial_error_index)
            self.__current_w = np.zeros(self.input_dimension + 1)

        return self.__current_w

    def run_and_show_histogram(self, repeated_times, max_update_times):
        w = self.__run(repeated_times, max_update_times)
        print('Q7: average number of updates:', w)
        # plt.hist(h)
        # plt.show()