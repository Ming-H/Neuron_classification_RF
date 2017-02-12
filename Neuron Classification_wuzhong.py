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
os.chdir("E:\data")


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

# Caculate right ratio
def get_ratio(x,y):
    sum = 0
    ratio = 0
    for i in range(len(x)):
        if x[i] == y[i]:
            sum += 1
    ratio = (sum * 1.0) / len(x)
    return ratio
    
def RF_model(data,L):
    X_train = pd.concat([data[0:L[0]],data[L[0]:L[1]],data[L[1]:L[2]],data[L[2]:L[3]],data[L[3]:L[4]]],ignore_index=True)
    print X_train
    '''
    X_test = data[L[0]:L[1]]
    

    #X_train = preprocessing.scale(X_train)   # 
    #X_test = preprocessing.scale(X_test)
    
    
    #clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    clf = RandomForestClassifier(n_estimators= 1000, min_samples_split=2, random_state = 0)
    
    #clf = RandomForestClassifier(n_estimators= 100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
    #min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True,
    #oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
    
    clf1 = clf.fit(X_train, y_train)
    result1 = clf1.predict(X_test)
    total_ratio1 = get_ratio(result1,y_test)
    print total_ratio1
    f.write("The test ratio is:"+str(total_ratio1)+"\n\n")
    
    
    X_mat = np.vstack((X_train,X_test))
    y_mat = y_train + y_test
   
    
    sum = 0
    num = 10
    all_ratio = 0
   
            
        # feature selection
    clf2 = ExtraTreesClassifier()                                                    #first feature select
    clf2 = clf2.fit(X_mat, y_mat) 
        #print clf2.feature_importances_ 
    model = SelectFromModel(clf2, prefit=True)
    X_new = model.transform(X_mat)
    print len(X_new)
    
    d = {}
    for k in range(1,X_new.shape[1]):
        X_train_new = SelectKBest(mutual_info_classif, k).fit_transform(X_new, y_mat)    #second feature select
        f.write(str(X_train_new[0])+"\n")
        clf2 = clf.fit(X_train_new[0:len(X_train)], y_train)    # predict
        result2 = clf2.predict(X_train_new[len(X_train)::])
        total_ratio2 = get_ratio(result2,y_test)
        f.write(str(total_ratio2)+"\n\n")
        d[k] = total_ratio2
    d1 = sorted(d.items(), key=operator.itemgetter(1),reverse=True) 
    print d1
    '''
   
if __name__=="__main__":
    f = open("Classification_differentanimals_results.txt","w")
    f.write(' '*13+"NEURON CLASSIFICATION WITH DIFFERENT ANIMALS RESULTS"+"\n\n")
    f.write("There 1791 neurons, 80% for training, others for testing."+'\n\n')    
    
    columns = ['Soma_Surface','N_stems','N_bifs','N_branch','N_tips','Width','Height','Depth','Type','Diameter','Diameter_pow','Length','Surface',
                 'SectionArea','Volume','EucDistance','PathDistance','Branch_Order','Terminal_degree','TerminalSegment','Taper_1','Taper_2',
                 'Branch_pathlength','Contraction','Fragmentation','Daughter_Ratio','Parent_Daughter_Ratio','Partition_asymmetry','Rall_Power','Pk',
                 'Pk_classic','Pk_2','Bif_ampl_local','Bif_ampl_remote','Bif_tilt_local','Bif_tilt_remote','Bif_torque_local','Bif_torque_remote',
                 'Last_parent_diam','Diam_threshold','HillmanThreshold','Helix','Fractal_Dim']
    
    data0 = pd.DataFrame(loadDataSet('monkey.txt'),columns=columns)        #408,326
    data1 = pd.DataFrame(loadDataSet('giraffe.txt'),columns=columns)       #350,280
    data2 = pd.DataFrame(loadDataSet('elegans.txt'),columns=columns)      #302,241
    data3 = pd.DataFrame(loadDataSet('chimpanzee.txt'),columns=columns)    #604,483
    data4 = pd.DataFrame(loadDataSet('cat.txt'),columns=columns)           #127,101       ## total:1791
    
    data0['flag'] = 0
    data1['flag'] = 1
    data2['flag'] = 2
    data3['flag'] = 3
    data4['flag'] = 4
    
    train_data = pd.concat([data0[0:int(0.8*data0.shape[0])],data1[0:int(0.8*data1.shape[0])],data2[0:int(0.8*data2.shape[0])],data3[0:int(0.8*data3.shape[0])],data4[0:int(0.8*data4.shape[0])]], ignore_index=True)
    test_data = pd.concat([data0[int(0.8*data0.shape[0]):data0.shape[0]],data1[int(0.8*data1.shape[0]):data1.shape[0]],data2[int(0.8*data2.shape[0]):data2.shape[0]],data3[int(0.8*data3.shape[0]):data3.shape[0]],data4[int(0.8*data4.shape[0]):data4.shape[0]]], ignore_index=True)
    print test_data

    
    #RF_model(data,L)
    f.close()
    print 'over'
    

    
    
