# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:16:06 2021

@author: arthurchien
"""

import csv
import numpy as np
with open('submission.csv') as csvfile:
    rows = csv.reader(csvfile)
    ans = list(rows)
    
with open('sample_submission.csv') as csvfile:
    rows = csv.reader(csvfile)
    new = list(rows)

new[1:10000] = ans[1:10000]
with open('submission1.csv','w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(new)