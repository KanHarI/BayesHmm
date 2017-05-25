# -*- coding: utf-8 -*-
"""
Created on Tue May 23 19:40:28 2017

@author: KanHar
"""

import datetime

class Sample(object):
    def __init__(self, date : datetime.date, openVal : float, closeVal : float):
        self.date = date
        self.open = openVal
        self.close = closeVal

class CsvParser(object):
    def __init__(self, file_path : str):
        self.lines = open(file_path, "rb").readlines()
    
    
    def items(self):
        for line in self.lines:
            lineParts = line[:-1].split(b',')
            rawDate = lineParts[0]
            year = int(rawDate[:4])
            month = int(rawDate[4:6])
            day = int(rawDate[6:8])
            date = datetime.date(year, month, day)
            openVal = float(lineParts[2])
            closeVal = float(lineParts[5])
            yield Sample(date, openVal, closeVal)
