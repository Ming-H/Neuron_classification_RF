# -*- coding: utf-8 -*-
"""
Created on Thurs May 26 15:28:03 2016

@author: HM
"""

print(__doc__)
import os
os.chdir("G:/python/neuron classification")
import pandas as pd
data = pd.read_csv('nrn-neuromorpho-ids.csv')
print data
file = open('index.txt','w' )
num = 0
for i in range(len(data)):
    file.write(data.id[i]+',')
    if num > 10:
        num = 0
        file.write('\n\n')
    else:
        num += 1
    
file.close
print len(data)  ##241