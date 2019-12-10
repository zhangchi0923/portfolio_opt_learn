import pandas as pd
import os
from dateutil.parser import *
import datetime
path = 'C://Users//10993//Documents//富国实习//富国实习//组合优化//组合优化数据1118//最终结果'

def concat_data(path):
    fileDir = os.listdir(path)
    result = pd.Series([])
    for file in fileDir:
        ret = pd.read_csv(path+'//'+file, index_col=[0])*0.01
        date = parse(ret.columns[0]).strftime('%Y-%m-%d')
        date_idx = [date]*ret.shape[0]
        ret.index = [date_idx, ret.index]
        ret.index.name = ['date', 'stock_code']
        ret.columns = ['score']
        if result.empty:
            result = ret
        else:
            result = pd.concat([result, ret], axis=0)
    return result



r = concat_data(path)
