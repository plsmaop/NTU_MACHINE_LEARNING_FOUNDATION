#-*- coding: utf-8 -*-
import time
import random
import numpy as np

class PLA():
    def __init__(self, file_path, input_dimension):
        """
            read input data from the given file path
        """
        self.input_dimension = input_dimension
        self.X, self.Y = self.load_data_from_file(file_path, input_dimension)

    @staticmethod
    def load_data_from_file(file_path, input_dimension):
        with open(file_path, 'r') as file:
            raw_data = file.readlines()
            data_num = len(raw_data)

            # add X0
            X = np.zeros((data_num, input_dimension + 1))
            Y = np.zeros(data_num)
            for ind, line in enumerate(raw_data):
                # X0 = 1
                X[ind, 0] = 1.0

                elements = line.strip().split()
                Y[ind] = int(elements[-1])
                for i in range(1, input_dimension + 1):
                    X[ind, i] = elements[i-1]
            return X, Y

    def __run(self, cycle, is_exceed_max_update, update_w):
        """
            arg:
                learning_rate: int
                cycle: [int]
                is_exceed_max_update: func return bool
                update_w: func([int], [int], int) return [int]
        """

        # initial w = 0
        w = np.zeros(self.input_dimension + 1)

        cycle_len = len(cycle)
        iteration = 0
        no_error_count = 0
        while not is_exceed_max_update():
            cycle_num = cycle[iteration]
            x = self.X[cycle_num]
            y = self.Y[cycle_num]

            dot_value = np.dot(x.T, w)
            if self.sign(dot_value) != y:
                # update w
                w = update_w(w, x, y)
                no_error_count = 0
            else:
                iteration = (iteration + 1) % cycle_len
                no_error_count += 1

            if no_error_count == cycle_len : break

        return w

    def run_naive(self, is_exceed_max_update, update_w):
        """
            naive PLA
        """
        return self.__run([i for i in range(len(self.X))], is_exceed_max_update, update_w)

    def run_with_random_cycle(self, is_exceed_max_update, update_w):
        """
            random cycle PLA
        """

        x_len = len(self.X)
        # set new seed
        random.seed(time.time())
        random_cycle = random.sample(range(x_len), x_len)
        return self.__run(random_cycle, is_exceed_max_update, update_w)

    @staticmethod
    def sign(num):
        # sign(0) = −1
        return 1 if (num > 0) else -1
