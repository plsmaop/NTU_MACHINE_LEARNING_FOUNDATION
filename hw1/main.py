#-*- coding: utf-8 -*-
from Q6 import Q6

repeated_time = 1126
q6_data = 'hw1_6_train.dat'

if __name__ == '__main__':
    q6 = Q6(q6_data, 4)
    q6.run_and_show_histogram(repeated_time)
