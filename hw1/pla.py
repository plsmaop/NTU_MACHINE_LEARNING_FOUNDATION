#-*- coding: utf-8 -*-
import random
import numpy as np

class PLA():
    def __init__(self, file_path, input_dimension):
        """
            read input data from the given file path
        """

        file = open(file_path, 'r')
        raw_data = file.readlines()
        data_num = len(raw_data)

        self.input_dimension = input_dimension

        # add X0
        self.X = np.zeros((data_num, input_dimension + 1))
        self.Y = np.zeros((data_num, 1))
        for ind, line in enumerate(raw_data):
            # X0 = 1
            self.X[ind, 0] = 1.0

            elements = line.strip().split()
            self.Y[ind, 0] = int(elements[-1])
            for i in range(1, input_dimension + 1):
                self.X[ind, i] = elements[i-1]
        
        file.close()

    def __run(self, learning_rate, cycle, fn):
        """
            arg:
                learning_rate: int
                cycle: [int]
                fn: function
        """

        if not learning_rate:
            learning_rate = 1.0

        # initial w = 0
        w = np.zeros(self.input_dimension + 1)

        cycle_len = len(cycle)
        iteration = 0
        while iteration < cycle_len:
            cycle_num = cycle[iteration]
            x = self.X[cycle_num]
            y = self.Y[cycle_num ,0]

            dot_value = np.dot(x.T, w)
            if self.sign(dot_value) != y:
                # update w
                w = w + learning_rate * y * x

                # custom function
                if fn:
                    fn()

                iteration = 0
            else:
                iteration += 1

        return w

    def run_naive(self, learning_rate, fn):
        """
            naive PLA
        """
        return self.__run(learning_rate, [i for i in range(len(self.X))], fn)

    def run_with_random_cycle(self, learning_rate, fn):
        """
            random cycle PLA
        """

        x_len = len(self.X)
        random_cycle = random.sample(range(x_len), x_len)
        return self.__run(learning_rate, random_cycle, fn)

    @staticmethod
    def sign(num):
        # sign(0) = −1
        return 1 if (num > 0) else -1
