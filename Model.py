# Python 3.6

import numpy as np

class Model(object):
    """An HMM model + bayesian interference for sequence prediction"""
    
    def __init__(self, number_of_states : int, number_of_outputs : int):
        self.numberOfStates = number_of_states
        self.numberOfOutputs = number_of_outputs
        self.STS = np.random.random((number_of_states, number_of_states)) + 0.5 # state to state transitions
        self.STO = np.random.random((number_of_states, number_of_outputs)) + 0.5 # state to output transitions
        # +0.5 followed by normalization caps outputs to range [0.25, 0.75]
        
        self.renormalize(self.STS)
        self.renormalize(self.STO)
        
        self.SP = np.zeros((self.numberOfStates,)) #State probabilities
        self.SP[0] = 1 # in the beginning, there is a probability of 1 for the first state
        
        self.dSPbydSTS = np.zeros((self.numberOfStates, self.numberOfStates, self.numberOfStates))
        self.dSPbydSTO = np.zeros((self.numberOfStates, self.numberOfStates, self.numberOfOutputs))
        
        self.dErrbydSTS = np.zeros((self.numberOfStates, self.numberOfStates))
        self.dErrbydSTO = np.zeros((self.numberOfStates, self.numberOfOutputs))
        
        self.outputs = None

    def renormalize(self, a : np.array):
        """Divides rows by sum (each row's sum = 1 after renormalization)"""
        a /= (a.sum(axis=1)[:,None])

    def advanceStates(self):
        """Move states one step forward and keeps track of gradients"""
        # indirect component
        self.dSPbydSTS = np.einsum("ij, ikl -> jkl", self.STS, self.dSPbydSTS)
        self.dSPbydSTO = np.einsum("ij, ikl -> jkl", self.STS, self.dSPbydSTO)
        # add direct derivative
        # TODO: find more efficient way to write following line?
        i,j,k = np.indices(self.dSPbydSTS.shape)
        self.dSPbydSTS[i==k] += np.outer(np.ones((self.numberOfStates,)), self.SP).reshape((self.numberOfStates*self.numberOfStates,))
        self.SP = self.SP.dot(self.STS) # State transtions
        self.outputs = None
    
    def sendRealOutputs(self, real_outs : np.array):
        """Sends the real inputs into the system to allow updating of the predicting data and gradient calculation"""
        if self.outputs is None:
            self.genOutputs()
        
        dErrbydO = 2*(real_outs - self.outputs)
        self.dErrbydSTO += np.einsum("i, j -> ij", self.SP, dErrbydO)
        dErrbydS = np.einsum("ij, j -> i", self.STO, dErrbydO)
        self.dErrbydSTO += np.einsum("i, ijk -> jk", dErrbydS, self.dSPbydSTO)
        self.dErrbydSTS += np.einsum("i, ijk -> jk", dErrbydS, self.dSPbydSTS)
        
        fitnesses = self.STO.dot(real_outs)
        sfits = fitnesses*self.SP
        ssfits = sum(sfits)
        self.SP = sfits/sum(sfits)
        self.dSPbydSTO = np.einsum("ijk, i -> ijk", self.dSPbydSTO, self.SP)
        self.dSPbydSTS = np.einsum("ijk, i -> ijk", self.dSPbydSTS, self.SP)
        i,j,k = np.indices(self.dSPbydSTO.shape)
        # TODO: find more efficient way to write this
        for i in range(self.numberOfStates):
            for j in range(self.numberOfStates):
                for k in range(self.numberOfOutputs):
                    if i == j:
                        self.dSPbydSTO[i,j,k] += (ssfits-sfits[i]) * real_outs[k] / (ssfits ** 2)
                    else:
                        self.dSPbydSTO[i,j,k] -= sfits[i] * real_outs[k] / (ssfits ** 2)
    
    def genOutputs(self):
        """Generate next outputs"""
        if self.outputs is None:
            self.outputs = self.SP.dot(self.STO)
        return self.outputs
    
    def applyGradient(self, learning_rate : float):
        """Applies gradient multiplied by learning rate and changes transition probabilities"""
        self.STO += self.dErrbydSTO * learning_rate
        self.STS += self.dErrbydSTS * learning_rate
        self.STO[self.STO > 1] = 1
        self.STO[self.STO < 0] = 0
        self.renormalize(self.STO)
        self.renormalize(self.STS)
    
    def reset(self):
        """resets flowing data"""
        self.SP *= 0
        self.SP[0] = 1
        self.dErrbydSTO *= 0
        self.dErrbydSTS *= 0
        self.dSPbydSTO *= 0
        self.dSPbydSTS *= 0
