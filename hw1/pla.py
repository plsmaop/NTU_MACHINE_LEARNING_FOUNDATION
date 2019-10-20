#-*- coding: utf-8 -*-
import numpy as np


class PLA():
    def __init__(self, file_path, input_dimension):
        """
            read input data from the given file path
        """

        file = open(file_path, 'r')
        raw_data = file.readlines()
        data_num = len(raw_data)

        self.X = np.zeros((data_num, input_dimension + 1))
        self.Y = np.zeros((data_num, 1))
        for ind, line in enumerate(raw_data):
            # X0 = 1
            self.X[ind, 0] = 1.0

            elements = line.strip().split()
            self.Y[ind, 0] = elements[-1]
            for i in range(1, input_dimension + 1):
                self.X[ind, i] = elements[i-1]
        
        file.close()