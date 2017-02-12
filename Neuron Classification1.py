# -*- coding: utf-8 -*-
"""
Created on Thurs May 26 15:28:03 2016

@author: HM
"""

print(__doc__)

import pandas as pd
import numpy as np
from pandas import DataFrame,Series
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB 
from sklearn.linear_model import LogisticRegression

# Generate data 
import os
#rz
os.chdir("F:/neuron classification/data")
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
X_test2 = pd.read_csv("X_test2.csv")
data1 = [2,2,2,2,2,5,5,5,1,1,1,1,1,3,3,3,3,3,3,3,3,3,4,5,5,5,7,7,7,7,7,7,7,4,4,4,4,6,6,6,6,6,6,6] 
data2 = [2,4,1,3,5,6,7]  
data3 = [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,7,7,7,7,7,7,7,7,7,7]  
y_train = np.array(data1)
y_test = np.array(data2)                                                  
y_test2 = np.array(data3) 
print len(data3)

print "There are 44 train datas\n"

# Caculate right ratio
def fn(x,y):
    sum = 0
    ratio = 0
    for i in range(len(x)):
        if x[i] == y[i]:
            sum += 1
    ratio = (sum * 1.0) / len(x)
    return ratio
 
    
# Fit classifier with RandomForestClassifier
print "***************************************************************************\n"
print "The result of RandomForestClassifier: "
#clf = tree.DecisionTreeClassifier(criterion='entropy')
#clf = tree.DecisionTreeClassifier(criterion='gini')
#clf = svm.LinearSVC()
#clf = LogisticRegression(penalty='l2') 
'''
clf = clf.fit(X_train, y_train)
result1 = clf.predict(X_test) 
print result1
print "The right ratio is", fn(result1,data2), "\n" 
result2 = clf.predict(X_test2)  
print result2
print "The right ratio is",fn(result2,data3), "\n" 


index = []
values = []
for num in range(10,1001,5):
    index.append(num)

    clf = RandomForestClassifier(n_estimators=num, criterion='entropy', max_depth=None, min_samples_split=2, 
         min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, 
         bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)   
    clf = clf.fit(X_train, y_train)
    result2 = clf.predict(X_test2)
    values.append(fn(result2,data3))
    
data_array = pd.DataFrame(values,index=index,columns=['c-rate'])
data_array.to_csv('canshu.txt')
'''
L = []

for item in range(2,4):
    clf = RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=None, min_samples_split=item, 
         min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, 
         bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
    clf = clf.fit(X_train, y_train)
    result2 = clf.predict(X_test2)
    L.append(fn(result2,data3))
data = pd.DataFrame(L)
data.columns=['min_samples_split']
print data