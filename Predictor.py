
import copy
import itertools

import numpy as np

from Model import Model
from CsvParser import  CsvParser
from VectTranslator import VectTranslator

class Predictor(object):
    def __init__(self, file_pathes : list, model : Model):
        parsed_files = map(CsvParser, file_pathes)
        self._itemLists = map(lambda csvFile: csvFile.items(), parsed_files)
        self._filterInputs()
        self._outToVect = VectTranslator()
        self._model = model

    def _filterInputs(self):
        # copy list immutable iteration while mutatuing
        for lst in copy.deepcopy(self._itemLists):
            target_date = next(lst).date
            self._itemLists = map(
                lambda sampList: filter(lambda item: item.date >= target_date, sampList),
                self._itemLists
            )

    def getSamples(self):
        samples_tpls = zip(*self._itemLists)
        for sample_tpl in samples_tpls:
            sample_tpl = map(lambda sample: (sample.open, sample.close), sample_tpl)
            sample_tpl = self._outToVect.translateToVector(sample_tpl)
            yield  sample_tpl


    def train(self, iterations : int, trace : bool = False, initial_learning_rate : float = 1e-30):
        realOutsOrig = list(self.getSamples())
        learningRate = initial_learning_rate
        lastErr = 0
        for i in range(iterations):
            error = 0
            for realOuts in realOutsOrig:
                self._model.advanceStates()
                outs = self._model.genOutputs()
                self._model.sendRealOutputs(realOuts)
                deltas = realOuts - outs
                error += np.vectorize(lambda x: x * x)(deltas).sum()
            if error > lastErr: # We do not care about halfing the learning rate in the first iteration
                learningRate *= 0.5
            else:
               learningRate *= 1.2
            lastErr = error
            if trace:
                print("error is: " + str(error) + "\tlearn: " + str(learningRate) + "\tSTO grad: " + str(np.sqrt(np.mean(np.square(self._model.dErrbydSTO)))) + "\tSTS grad: " + str(np.sqrt(np.mean(np.square(self._model.dErrbydSTS)))))
            self._model.applyGradient(learningRate)
            self._model.reset()
