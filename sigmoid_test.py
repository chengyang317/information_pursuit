# coding=utf-8
__author__ = "Philip_Cheng"

import numpy as np


x = np.arange(0, 50, 0.1)

y = (2 / (1 + np.exp(-2 * x / 6)) - 1) * 6

print y