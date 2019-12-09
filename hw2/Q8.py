# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from decision_stump import DecisionStump


class Q8(DecisionStump):
    def __init__(self, data_size, experiment_times, noise_prob):
        super(Q7, self).__init__(data_size, experiment_times, noise_prob)
