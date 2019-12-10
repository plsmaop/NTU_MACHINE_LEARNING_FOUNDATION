# -*- coding: utf-8 -*-
from decision_stump import DecisionStump

if __name__ == '__main__':
    q7 = DecisionStump(20, 1000)
    q7.run_and_render_histogram()

    q8 = DecisionStump(2000, 1000)
    q8.run_and_render_histogram()
