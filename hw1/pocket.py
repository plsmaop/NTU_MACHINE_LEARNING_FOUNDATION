#-*- coding: utf-8 -*-
import numpy as np
from pla import PLA

class Pocket(PLA):
    def __init__(self, file_path, verify_set_path, input_dimension):
        super(Pocket, self).__init__(file_path, input_dimension)

        self.__test_X, self.__test_Y = PLA.load_data_from_file(verify_set_path, input_dimension)
        
    @staticmethod
    def get_error_index(X, Y, w):
        error_index = []
        for i, (x, y) in enumerate(zip(X, Y)):
            if PLA.sign(np.dot(x.T, w)) != y:
                error_index.append(i)

        return error_index

    def train(self, is_exceed_max_update, update_w):

        # w0 = 0
        w = np.zeros(self.input_dimension + 1)
        while not is_exceed_max_update():
            w = update_w(w, self.X, self.Y)

        return w