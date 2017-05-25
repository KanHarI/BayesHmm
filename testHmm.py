# -*- coding: utf-8 -*-
# Python 3.6
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

from Model import Model
from Predictor import Predictor

NUMBER_OF_STATES = 5
NUMBER_OF_OUTPUTS = 2
FILE_PATH = r"C:\Programming\Algo\quantquote_daily_sp500_83986\daily\table_aapl.csv"
FILE_PATH2 = r"C:\Programming\Algo\quantquote_daily_sp500_83986\daily\table_ge.csv"


m = Model(NUMBER_OF_STATES, NUMBER_OF_OUTPUTS)
p = Predictor([FILE_PATH, FILE_PATH2], m)
p.train(100, True, 1e-5)

# learningFactor = 1e-5
# error = 0
# lastErr = 0
#
# v = VectTranslator()
#
# for j in range(1000):
#     error = 0
#     appl = CsvParser(FILE_PATH)
#     for sample in appl.items():
#         m.advanceStates()
#         outs = m.genOutputs()
#         realOuts = v.translateToVector([(sample.open, sample.close)])
#         m.sendRealOutputs(realOuts)
#         deltas = realOuts - outs
#         error += np.vectorize(lambda x: x*x)(deltas).sum()
#     if error > lastErr:
#         learningFactor *= 0.5
#     else:
#         learningFactor *= 1.1
#     lastErr = error
#     print("error is: " + str(error) + " learning rate: " + str(learningFactor))
#     m.applyGradient(learningFactor)
#     m.reset()
