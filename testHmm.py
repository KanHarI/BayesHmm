# -*- coding: utf-8 -*-
# Python 3.6
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

from Model import Model

NUMBER_OF_STATES = 2
NUMBER_OF_OUTPUTS = 2



m = Model(NUMBER_OF_STATES, NUMBER_OF_OUTPUTS)
print(m.STS)
print(m.SP)
print(m.STO)

learningFactor = 0.01

for i in range(1000):
    for j in range(10):
        m.advanceStates()
        m.genOutputs()
        m.sendRealOutputs(np.array([1,0]))
        m.advanceStates()
        m.genOutputs()
        m.sendRealOutputs(np.array([0,1]))
    m.applyGradient(learningFactor)
    m.reset()
    m.advanceStates()
    print(m.genOutputs())
    m.sendRealOutputs(np.array([1,0]))
    m.advanceStates()
    print(m.genOutputs())
    m.sendRealOutputs(np.array([0,1]))
    m.advanceStates()
    print(m.genOutputs())
    m.reset()
print(m.genOutputs())
