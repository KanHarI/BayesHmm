
import copy

from Model import Model
from CsvParser import  CsvParser
from VectTranslator import VectTranslator

class Predictor(object):
    def __init__(self, filesPathes):
        parsedFiles = map(CsvParser, filesPathes)
        self._itemLists = map(lambda fileName: fileName.items(), parsedFiles)
        self._filterInputs()
        self._outToVect = VectTranslator()

    def _filterInputs(self):
        # copy list immutable iteration while mutatuing
        for lst in copy.deepcopy(self._itemLists):
            self._itemLists = map(
                lambda curDate: filter(lambda item: item.date >= curDate),
                next(lst).date
            )

    def getSamples(self):
        for openClosePairs in zip(self._itemLists):
            vect = self._outToVect(openClosePairs)
