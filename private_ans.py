# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 13:16:42 2021

@author: arthurchien
"""

import csv

with open('public.csv') as csvfile:
    rows = csv.reader(csvfile)
    public = list(rows)
    csvfile.close()
with open('submission.csv') as csvfile:
    rows = csv.reader(csvfile)
    new = list(rows)
    csvfile.close()
    
new[1:10001] = public[1:10001]

with open('submission.csv','w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(new)