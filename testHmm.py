# -*- coding: utf-8 -*-
# Python 3.6
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

NUMBER_OF_STATES = 4
NUMBER_OF_OUTPUTS = 2

class Model(object):
    
    def __init__(self, numberOfStates, numberOfOutputs):
        self.numberOfStates = numberOfStates
        self.numberOfOutputs = numberOfOutputs
        self.STS = np.random.random((numberOfStates, numberOfStates)) + 0.5 # state to state transitions
        self.STO = np.random.random((numberOfStates, numberOfOutputs)) + 0.5 # state to output transitions
        # +0.5 followed by normalization caps outputs to range [0.25, 0.75]
        
        self.renormalize(self.STS)
        self.renormalize(self.STO)
        
        self.SP = np.zeros((self.numberOfStates,)) #State probabilities
        self.SP[0] = 1 # in the beginning, there is a probability of 1 for the first state
        
        self.dSPbydSTS = np.zeros((self.numberOfStates, self.numberOfStates, self.numberOfStates))
        self.dSPbydSTO = np.zeros((self.numberOfStates, self.numberOfStates, self.numberOfOutputs))
        
        self.outputs = None

    def renormalize(self, a):
        """Divides rows by sum (each row's sum = 1 after renormalization)"""
        a /= (a.sum(axis=1)[:,None])

    def advanceStates(self):
        """Move states one step forward and keeps track of gradients"""
        # indirect component
        self.dSPbydSTS = np.einsum("ij, ikl -> jkl", self.STS, self.dSPbydSTS)
        # add direct derivative
        # TODO: find more efficient way to write following line?
        i,j,k = np.indices(self.dSPbydSTS.shape)
        self.dSPbydSTS[i==k] += np.outer(np.ones((self.numberOfStates,)), self.SP).reshape((self.numberOfStates*self.numberOfStates,))
        self.SP = self.SP.dot(self.STS) # State transtions
    
    def sendRealOutputs(self, realOuts):
        """Sends the real inputs into the system to allow updating of the predicting data and gradient calculation"""
        fitnesses = self.STO.dot(realOuts)
        tmpSum = sum(self.SP*fitnesses)
        fitnesses /= tmpSum # fitnesses as applied to State Probabilities
        self.SP *= fitnesses
        self.dSPbydSTS = np.einsum("ijk, i -> ijk", self.dSPbydSTS, fitnesses)
        # self.dSPbydSTO = # TODO:
        print(m.dSPbydSTS)
        # TODO: calculate fitness gradient
        
    
    def genOutputs(self):
        if self.outputs == None:
            self.outputs = self.SP.dot(self.STO)
        return self.outputs


m = Model(NUMBER_OF_STATES, NUMBER_OF_OUTPUTS)
print(m.STS)
print(m.SP)
print(m.STO)
m.advanceStates()
m.sendRealOutputs(np.array([1,0]))
