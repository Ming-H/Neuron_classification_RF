# -*- coding: utf-8 -*-
"""
Created on Thurs May 26 15:28:03 2016

@author: HM
"""
# http://scikit-learn.org/stable/modules/feature_selection.html
# http://www.cnblogs.com/jasonfreak/p/5448385.html

print(__doc__)

import os
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
os.chdir("F:/neuron classification/data")

# Caculate right ratio
def get_ratio(x,y):
    sum = 0
    ratio = 0
    for i in range(len(x)):
        if x[i] == y[i]:
            sum += 1
    ratio = (sum * 1.0) / len(x)
    return ratio
 
def get_result():
    columns = ['Soma_Surface','N_stems','N_bifs','N_branch','N_tips','Width','Height','Depth','Type','Diameter','Diameter_pow',
           'Length','Surface','SectionArea','Volume','EucDistance','PathDistance','Branch_Order','Terminal_degree','TerminalSegment',
           'Taper_1','Taper_2','Branch_pathlength','Contraction','Fragmentation','Daughter_Ratio','Parent_Daughter_Ratio',
           'Partition_asymmetry','Rall_Power','Pk_classic','Bif_ampl_local','Bif_ampl_remote','Bif_tilt_local','Bif_tilt_remote',
           'Bif_torque_local','Bif_torque_remote','Last_parent_diam','Diam_threshold','HillmanThreshold','Helix','Fractal_Dim']

    data = pd.read_csv("X_train_totla.csv")
    data.columns=columns
    X_train = data.values 
    f.write(str(X_train[0])+",")
    f.write(str(len(X_train[0]))+",")
    
    #X_train = preprocessing.scale(X_train)
    
    y_train = [2,2,2,2,2,5,5,5,1,1,1,1,1,3,3,3,3,3,3,3,3,3,4,5,5,5,7,7,7,7,7,7,7,4,4,
               4,4,6,6,6,6,6,6,6,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,
               3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,7,7,7,7,7,7,7,7,7,7]
    
    X_val = pd.read_csv("X_test.csv")
    X_val = preprocessing.scale(X_val)
    y_val = [2,4,1,3,5,6,7]
    
    
    clf = RandomForestClassifier(n_estimators= 500, min_samples_split=2, random_state= 0)
    #clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    
    '''
    clf = RandomForestClassifier(n_estimators= 420, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
    min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True,
    oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)    #0.85
    
    # validation result
    clf0 = clf.fit(X_train[0:44], y_train[0:44])
    result0 = clf0.predict(X_val)
    total_ratio0 = get_ratio(result0,y_val)
    f.write("The validation ratio is:"+str(total_ratio0)+"\n\n")
    '''
    # test result
    clf1 = clf.fit(X_train[0:44], y_train[0:44])
    result1 = clf1.predict(X_train[44::])
    total_ratio1 = get_ratio(result1,y_train[44::])
    f.write("The test ratio is:"+str(total_ratio1)+"\n\n")

    '''
    clf2 = ExtraTreesClassifier()
    clf2 = clf2.fit(X_train, y_train) 
    model = SelectFromModel(clf2, prefit=True)
    X_new = model.transform(X_train)
    f.write(str(X_new[0])+","+'\n')    
    X_train_new = SelectKBest(mutual_info_classif, 5).fit_transform(X_new, y_train)
    f.write(str(X_train_new[0])+",")
    
    clf2 = clf.fit(X_train_new[0:44], y_train[0:44])
    result2 = clf2.predict(X_train_new[44::])
    total_ratio2 = get_ratio(result2,y_train[44::])
    f.write('\n'+"The upgrade ratio is:"+str(total_ratio2)+'\n\n')
    
    sum = 0
    for i in range(10):
        # feature selection
        clf2 = ExtraTreesClassifier()
        clf2 = clf2.fit(X_train, y_train) 
        #print clf2.feature_importances_ 
        model = SelectFromModel(clf2, prefit=True)
        X_new = model.transform(X_train)
      
        d={}
        for k in range(1,X_new.shape[1]):
            X_train_new = SelectKBest(mutual_info_classif, k).fit_transform(X_new, y_train)
            clf2 = clf.fit(X_train_new[0:44], y_train[0:44])
            result2 = clf2.predict(X_train_new[44::])
            total_ratio2 = get_ratio(result2,y_train[44::])
            d[k] = total_ratio2 
        d1 = sorted(d.items(), key=operator.itemgetter(1),reverse=True) 
        f.write(str(d1[0:3])+"\n") 
        sum += d1[0][1]
    ave_ratio = float(sum)/10
    f.write('\n'+"The ave ratio is:"+str(ave_ratio)+"\n\n")
    f.write("The features are:"+'\n'+str(X_train[0])+"\n\n")
    
    # k =4
    all_ratio = 0
    for i in range(10):      #测量10次取平均值
        # feature selection
        clf2 = ExtraTreesClassifier()
        clf2 = clf2.fit(X_train, y_train) 
        #print clf2.feature_importances_ 
        model = SelectFromModel(clf2, prefit=True)
        X_new = model.transform(X_train)
       
        X_train_new = SelectKBest(mutual_info_classif, 4).fit_transform(X_new, y_train)
        f.write(str(X_train_new[0])+",")
        clf2 = clf.fit(X_train_new[0:44], y_train[0:44])
        result2 = clf2.predict(X_train_new[44::])
        total_ratio2 = get_ratio(result2,y_train[44::])
        f.write(str(total_ratio2)+"\n") 
        all_ratio += total_ratio2
    ave_ratio4 = float(all_ratio)/10
    f.write('\n'+"The ave ratio4 is:"+str(ave_ratio4)+'\n\n')
    return ave_ratio4
    '''
  
  
        
        

    
    
    
    
    
    
    
if __name__=="__main__":
    f = open("Classification_results.txt","w")
    f.write(' '*13+"NEURON CLASSIFICATION RESULTS"+"\n\n")
    f.write("There 104 neurons, 44 for training, others for testing."+'\n\n')    
    get_result()
    f.close()
    print 'over'
    
    
