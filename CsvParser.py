# -*- coding: utf-8 -*-
"""
Created on Tue May 23 19:40:28 2017

@author: KanHar
"""

class CsvParser(object):
    def __init__(self, filePath):
        self.lines = open(filePath, "rb").readlines()
    
    
    def getItems(self):
        for line in self.lines:
            lineParts = line[:-1].split(b',')
