# -*- coding: utf-8 -*-
"""
Created on Thurs May 26 15:28:03 2016

@author: HM
"""
# http://scikit-learn.org/stable/modules/feature_selection.html
# http://www.cnblogs.com/jasonfreak/p/5448385.html

print(__doc__)


import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, SelectFpr, SelectFwe, SelectFdr, GenericUnivariateSelect
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import operator 
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
import os
os.chdir("E:\data\data-3\swc")


def loadDataSet(fileName):   
    fr = open(fileName)
    L = []
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        if len(curLine)>6:
            L.append(float(curLine[2]))
    dataMatrow = np.array(L)
    dataMat = dataMatrow.reshape((-1,43))
    return np.nan_to_num(dataMat)    #preprocessing.scale(X)

    
def get_data(data1,data2,data3,data4,data5):
    L1 = [data2[0:0.8*len(data2)],data3[0:0.8*len(data3)],data4[0:0.8*len(data4)],data5[0:0.8*len(data5)]]
    datatrain = data1[0:0.8*len(data1)]
    for i in L1:
        datatrain = np.vstack((datatrain,i))     #1431  
     
    L2 = [data2[0.8*len(data2)::],data3[0.8*len(data3)::],data4[0.8*len(data4)::],data5[0.8*len(data5)::]]
    datatest = data1[0.8*len(data1)::]
    for i in L2:
        datatest = np.vstack((datatest,i))       #360
    
    train_label = []
    L3 = [data1[0:0.8*len(data1)].shape[0],data2[0:0.8*len(data2)].shape[0],data3[0:0.8*len(data3)].shape[0],data4[0:0.8*len(data4)].shape[0],data5[0:0.8*len(data5)].shape[0]]
    for i in range(len(L3)):
        for j in range(L3[i]):
            train_label.append(i)
    
    test_label = []
    L4 = [data1[0.8*len(data1)::].shape[0],data2[0.8*len(data2)::].shape[0],data3[0.8*len(data3)::].shape[0],data4[0.8*len(data4)::].shape[0],data5[0.8*len(data5)::].shape[0]]
    for i in range(len(L4)):
        for j in range(L4[i]):
            test_label.append(i)   
            
    return datatrain,datatest,train_label,test_label

    
# Caculate right ratio
def get_ratio(x,y):
    sum = 0
    ratio = 0
    for i in range(len(x)):
        if x[i] == y[i]:
            sum += 1
    ratio = (sum * 1.0) / len(x)
    return ratio
    

def RF_model(data):
    X_train = data[0]
    X_test = data[1]
    y_train = data[2]
    y_test = data[3]

    #X_train = preprocessing.scale(X_train)   # 
    #X_test = preprocessing.scale(X_test)
    
    
    #clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    #clf = RandomForestClassifier(n_estimators= 1000, min_samples_split=1, random_state = 0)
    
    clf = RandomForestClassifier(n_estimators= 1000, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
    min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True,
    oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
    
    clf1 = clf.fit(X_train, y_train)
    result1 = clf1.predict(X_test)
    total_ratio1 = get_ratio(result1,y_test)
    print total_ratio1
    f.write("The test ratio is:"+str(total_ratio1)+"\n\n")
    
    
    X_mat = np.vstack((X_train,X_test))
    y_mat = y_train + y_test
   
    
    k = 5
    num = 10
    
    all_ratio = 0
    for i in range(num):
            
        # feature selection
        clf2 = ExtraTreesClassifier()                                                    #first feature select
        clf2 = clf2.fit(X_mat, y_mat) 
        #print clf2.feature_importances_ 
        model = SelectFromModel(clf2, prefit=True)
        X_new = model.transform(X_mat)
            
        X_train_new = SelectKBest(mutual_info_classif, k).fit_transform(X_new, y_mat)    #second feature select
        f.write(str(X_train_new[0])+",")
        
        clf2 = clf.fit(X_train_new[0:len(X_train)], y_train)    # predict
        result2 = clf2.predict(X_train_new[len(X_train)::])
        total_ratio2 = get_ratio(result2,y_test)
        f.write(str(total_ratio2)+"\n")
        all_ratio += total_ratio2
    ave_ratio = float(all_ratio)/num
    print ave_ratio
    f.write(str(ave_ratio)+"\n") 
 
   
if __name__=="__main__":
    f = open("Classification_differenttypes_results.txt","w")
    f.write(' '*13+"NEURON CLASSIFICATION WITH DIFFERENT ANIMALS RESULTS"+"\n\n")
    f.write("There 241 neurons, 80% for training, others for testing."+'\n\n')    
    
    data1 = loadDataSet('monkey.txt')        #408,326
    data2 = loadDataSet('giraffe.txt')       #350,280
    data3 = loadDataSet('elegans.txt')       #302,241
    data4 = loadDataSet('chimpanzee.txt')    #604,483
    data5 = loadDataSet('cat.txt')           #127,101       ## total:1791
    
    data = get_data(data1,data2,data3,data4,data5)
  
    RF_model(data)
    
    f.close()
    print 'over'
    

    
    
