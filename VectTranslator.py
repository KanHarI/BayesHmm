# -*- coding: utf-8 -*-
"""
Created on Tue May 23 21:25:15 2017

@author: Itaykh
"""

import numpy as np

class VectTranslator(object):
    def __init__(self, exponent=10):
        self.exponent = exponent
    
    def translateToVector(self, inTuples : list):
        vector = []
        for tpl in inTuples:
            sm = tpl[0]+tpl[1]
            v0 = (tpl[0]/sm)**self.exponent
            v1 = (tpl[1]/sm)**self.exponent
            vector.append(v0)
            vector.append(v1)
        vector = np.array(vector)
        vector = vector/vector.sum()
        return vector
