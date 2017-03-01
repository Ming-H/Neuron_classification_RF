# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import MySQLdb
from sqlalchemy import create_engine
import time
import re


    
if __name__=="__main__":
    start = time.clock()
    conn = MySQLdb.connect(host='localhost',port = 3306,user='haoming',passwd='111111',db ='neuron',charset='utf8')
    cur = conn.cursor()
    engine =create_engine('mysql+mysqldb://haoming:111111@localhost:3306/neuron?charset=utf8')
    sql = "select neuron_name,neuron_features,total_Sum from hippocampal"
    data = pd.read_sql(sql,conn)
    data = data.set_index(['neuron_name','neuron_features'])
    data = data.unstack()
    data = data.reset_index('neuron_name')
    columns ='neuron_name'+ data.columns.levels[1]
    print data.columns.levels[1]
    #data.to_sql('all_cells_scaled_total_Sum',engine,if_exists='replace',index=False,chunksize=1000)    
    end = time.clock()
    print end-start
    