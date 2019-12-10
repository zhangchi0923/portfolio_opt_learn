import numpy as np
import pandas as pd
from dataPrep import *
from opt_mod import *
#设置路径

path = '/Users/zhangchi/富国实习/组合优化/优化器'
inpath = path+'/Input'
outpath = path+'/Output'

def change_format(data):
    data.index = pd.to_datetime(data.index.astype(str))
    data = np.transpose(data).unstack()     # df -> series
    return data


def filter_data(data):
    price = pd.read_csv(open(path + '/Input/复权收盘价.csv'), index_col=[0])
    price = change_format(price).sort_index().to_frame('月末收盘价')

    state = pd.read_csv(open(path + '/Input/是否在市.csv'), index_col=[0])
    state = change_format(state).sort_index().to_frame('state')

    listlen = pd.read_csv(open(path + '/Input/上市天数.csv'), index_col=[0])
    listlen = change_format(listlen).to_frame('上市天数')

    ST = pd.read_csv(open(path + '/Input/特殊处理.csv'), index_col=[0])
    ST = change_format(ST).to_frame('特殊处理').fillna(1)

    zhandie = pd.read_csv(open(path + '/Input/涨跌停.csv'), index_col=[0])
    zhandie = change_format(zhandie).to_frame('涨跌停')

    data = pd.concat([data, price, state, listlen, ST, zhandie], join_axes=[data.index], axis=1)

    data = data.query('state == 1 ')
    data = data.query('上市天数 >= 180 ')
    data = data.query('特殊处理 == 1 ')
    data = data.query('涨跌停!= 1 ')
    data = data.dropna()
    data.index.names = ['date', 'stcode']
    return data

stockPool = get_stockPool()
mu = get_mu('2018-01-31')

# 3.市值
size = pd.read_csv(inpath + '/流通市值.csv', index_col=[0])
size = change_format(size)

# 4、中信一级行业
industry = pd.read_csv(inpath + '/中信一级行业.csv', index_col=[0], encoding='gbk')
industry = change_format(industry)

# 5、基准指数
# base_weight = pd.read_csv(inpath + '/中证500成份权重.csv', index_col=[0])
base_weight = pd.read_csv(inpath + '/沪深300成份权重.csv', index_col=[0])
base_weight = change_format(base_weight).fillna(0)

stock_data = pd.concat([stockPool, mu, size, industry, base_weight], join_axes=[stockPool.index], axis=1)
stock_data.columns = ['指数权重', '预期收益', '流通市值', '行业', '基准指数权重']

# 加入交易条件限制，即剔除不满足条件的股票
stock_data = filter_data(stock_data)

stock_data_M = stock_data.loc['2018-01-31']


mu = stock_data_M['预期收益'].values.tolist()
wb = stock_data_M['基准指数权重'].values.tolist()
industry = pd.get_dummies(stock_data_M['行业']).T.values.tolist()
size = stock_data_M['流通市值'].values.tolist()
gamma = 15
delta = 0.01 / np.sqrt(12)
sizeLimit = 0.1
industryLimit = 0.05

stock_list = stock_data_M.index.get_level_values(1)
Cov = get_cov_estimator(stock_list, '2018-01-31', window=252)
n = len(Cov)
cov = Cov.copy().tolist()

GT = np.linalg.cholesky(cov)

w = MarkowitzWithRisk(n, mu, wb, GT, gamma, delta, industry, size, industryLimit, sizeLimit)
