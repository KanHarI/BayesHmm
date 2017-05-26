# -*- coding: utf-8 -*-
"""
Created on Tue May 23 19:40:28 2017

@author: KanHar
"""

import datetime

class Sample(object):
    def __init__(self, date : datetime.date, open_val : float, close_val : float):
        self.date = date
        self.open = open_val
        self.close = close_val

class CsvParser(object):
    def __init__(self, file_path : str):
        self.lines = open(file_path, "rb").readlines()
    
    
    def items(self):
        for line in self.lines:
            line_parts = line[:-1].split(b',')
            raw_date = line_parts[0]
            year = int(raw_date[:4])
            month = int(raw_date[4:6])
            day = int(raw_date[6:8])
            date = datetime.date(year, month, day)
            open_val = float(line_parts[2])
            close_val = float(line_parts[5])
            yield Sample(date, open_val, close_val)
