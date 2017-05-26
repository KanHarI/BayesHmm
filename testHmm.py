# -*- coding: utf-8 -*-
# Python 3.6
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

from Model import Model
from Predictor import Predictor

NUMBER_OF_STATES = 10
NUMBER_OF_OUTPUTS = 4
FILE_PATH = r"C:\Programming\Algo\quantquote_daily_sp500_83986\daily\table_aapl.csv"
FILE_PATH2 = r"C:\Programming\Algo\quantquote_daily_sp500_83986\daily\table_ge.csv"



m = Model(NUMBER_OF_STATES, NUMBER_OF_OUTPUTS)

print("STS")
print(m.STS)
print("STO")
print(m.STO)

p = Predictor([FILE_PATH, FILE_PATH2], m)
p.train(5000, True, 1e-3)

print("STS")
print(m.STS)
print("STO")
print(m.STO)
