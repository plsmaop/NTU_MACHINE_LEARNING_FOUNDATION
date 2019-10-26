#-*- coding: utf-8 -*-
from Q6 import Q6
from Q7 import Q7

repeated_times = 1126
q6_data = 'hw1_6_train.dat'
q7_data = 'hw1_7_train.dat'
q7_test = 'hw1_7_test.dat'

max_update_times = 100

if __name__ == '__main__':
    q6 = Q6(q7_data, 4)
    q6.run_and_show_histogram(repeated_times)

    q7 = Q7(q7_data, q7_test, 4)
    # q7.run_and_show_histogram(repeated_times, max_update_times)