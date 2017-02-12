# -*- coding: utf-8 -*-
"""
Created on Thurs May 26 15:28:03 2016

@author: HM
"""
# http://scikit-learn.org/stable/modules/feature_selection.html
# http://www.cnblogs.com/jasonfreak/p/5448385.html
# http://blog.csdn.net/bea_tree/article/details/50757338
print(__doc__)

import os
import numpy as np
import pandas as pd
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import pandas as pd   
import warnings   
warnings.filterwarnings("ignore")  
import seaborn as sns  
import matplotlib.pyplot as plt
from pandas.tools.plotting import radviz   
sns.set(style="white", color_codes=True) 
os.chdir("F:/neuron classification/data")

columns = ['Soma_Surface','N_stems','N_bifs','N_branch','N_tips','Width','Height','Depth','Type','Diameter','Diameter_pow',
           'Length','Surface','SectionArea','Volume','EucDistance','PathDistance','Branch_Order','Terminal_degree','TerminalSegment',
           'Taper_1','Taper_2','Branch_pathlength','Contraction','Fragmentation','Daughter_Ratio','Parent_Daughter_Ratio',
           'Partition_asymmetry','Rall_Power','Pk_classic','Bif_ampl_local','Bif_ampl_remote','Bif_tilt_local','Bif_tilt_remote',
           'Bif_torque_local','Bif_torque_remote','Last_parent_diam','Diam_threshold','HillmanThreshold','Helix','Fractal_Dim']

data = pd.read_csv("X_train_totla.csv")
data.columns=columns
data0 = data.loc[:,[u'N_stems', u'N_tips', u'Type', u'Taper_1', u'Bif_torque_remote']]
y_train = [2,2,2,2,2,5,5,5,1,1,1,1,1,3,3,3,3,3,3,3,3,3,4,5,5,5,7,7,7,7,7,7,7,4,4,
               4,4,6,6,6,6,6,6,6,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,
               3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,7,7,7,7,7,7,7,7,7,7]
y_array = pd.DataFrame(y_train,columns=['flag'])
data1 = pd.concat([data0, y_array],axis=1)
index = pd.DataFrame(data1.index)
index.columns=['ID']
data2 = pd.concat([index,data1],axis=1)


mpl.rcParams['font.size'] = 17
plt.xticks(fontsize = 17)
plt.yticks(fontsize = 17)
plt.xlabel('n_estimators',fontsize = 17)
plt.ylabel('accuracy',fontsize = 17)
plt.legend(loc='best')  

sns.boxplot(x="flag", y="N_stems", data=data2)
sns.boxplot(x="flag", y="N_tips", data=data2)
sns.boxplot(x="flag", y="Type", data=data2)
sns.boxplot(x="flag", y="Taper_1", data=data2)
sns.boxplot(x="flag", y="Bif_torque_remote", data=data2)



sns.pairplot(data2.drop("ID", axis=1), hue="flag", size=3) 
sns.pairplot(data2.drop("ID", axis=1), hue="flag", size=3, diag_kind="kde")

 
radviz(data2.drop("ID", axis=1), "flag") 