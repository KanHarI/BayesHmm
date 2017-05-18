# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

NUMBER_OF_STATES = 3
NUMBER_OF_OUTPUTS = 2

class Model(object):
    
    def __init__(self, numberOfStates, numberOfOutputs):
        self.STS = np.random.random((numberOfStates, numberOfStates)) + 0.5 # state to state transitions
        self.STO = np.random.random((numberOfStates, numberOfOutputs)) + 0.5 # state to output transitions
        # +0.5 followed by normalization caps outputs to range [0.25, 0.75]
        
        self.renormalize(self.STS)
        self.renormalize(self.STO)
        
        self.SP = np.zeros((NUMBER_OF_STATES,)) #State probabilities
        self.SP[0] = 1 # in the beginning, there is a probability of 1 for the first state
        
        self.dSPbydSTS = np.zeros((NUMBER_OF_STATES, NUMBER_OF_STATES, NUMBER_OF_STATES))

    def renormalize(self, a):
        a /= (a.sum(axis=1)[:,None])

    def advanceStates(self):
        print m.dSPbydSTS
        print "AAA"
        m.dSPbydSTS = np.einsum("ij, ikl -> jkl", m.STS, m.dSPbydSTS)
        print m.dSPbydSTS
        print "AAB"
        print m.dSPbydSTS
        print "BBB"
        # add direct derivative
        i,j,k = np.indices(m.dSPbydSTS.shape)
        m.dSPbydSTS[i==k] += np.outer(np.ones((3,)), m.SP).reshape((9,))
        print "CCC"
        for i in xrange(self.SP.size):
            pass
        self.SP = self.SP.dot(self.STS) # State transtions
    
    def genOutputs(self):
        return self.SP.dot(self.STO)


m = Model(NUMBER_OF_STATES, NUMBER_OF_OUTPUTS)
print m.STS
print m.SP
#print m.STO
m.advanceStates()
#print "\nStates update\n"
#print m.SP
m.advanceStates()
#print m.genOutputs()
#m.advanceStates()
print "\nStates update\n"
print m.SP
#print m.genOutputs()
print "==="
print m.dSPbydSTS
