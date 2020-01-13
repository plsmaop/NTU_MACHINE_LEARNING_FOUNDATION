# -*- coding: utf-8 -*-
from logistic_regression import LogisticRegression
from gradient_descent import GradientDescent, StocasticGradientDescent

HW3_TRAIN = 'hw3_train.dat'
HW3_TEST = 'hw3_test.dat'

if __name__ == '__main__':
    zero_zero_point_one = LogisticRegression(
        0.01, 2000, GradientDescent(), StocasticGradientDescent())
    zero_zero_point_one.caculate_and_plot(HW3_TRAIN, HW3_TEST)

    zero_zero_zero_point_one = LogisticRegression(
        0.001, 2000, GradientDescent(), StocasticGradientDescent())
    zero_zero_zero_point_one.caculate_and_plot(HW3_TRAIN, HW3_TEST)
