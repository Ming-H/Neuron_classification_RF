# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import MySQLdb
from sqlalchemy import create_engine
import time
import os
#os.chdir("F:/project/2 neuron classification/dataset/Zhao 2014 ArXiv/All_Cells_Scaled/1features")
os.chdir("F:/project/2 neuron classification/dataset/Lu 2015 Neuroinformatics/Pyramidal Neurons 434/features")

def eachFile(filepath):
    pathDir = os.listdir(filepath)
    data = pd.DataFrame()
    for item in pathDir:
        print type(item)
        new_data = pd.read_csv(item,sep='\t',header=None)
        new_data['group'] = item[:-4]
        data = pd.concat([data,new_data],ignore_index=True)
        print data.shape  
    data.columns=['neuron_name','neuron_features','total_Sum','compartments','compartments','minimum','average','maximum','S_D','groupID']
    return data
    
    
if __name__=="__main__":
    start = time.clock()
    conn = MySQLdb.connect(host='localhost',port = 3306,user='haoming',passwd='111111',db ='neuron',charset='utf8')
    cur = conn.cursor()
    engine =create_engine('mysql+mysqldb://haoming:111111@localhost:3306/neuron?charset=utf8')
    
    filepath = 'F:/project/2 neuron classification/dataset/Lu 2015 Neuroinformatics/Pyramidal Neurons 434/features'
    data = eachFile(filepath)
    data.to_sql('pyramidal',engine,if_exists='replace',index=False,chunksize=1000)    
    
    end = time.clock()
    print end-start
    