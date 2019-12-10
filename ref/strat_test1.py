# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['SimHei']

## 优化框架
from optimizer import *

# 计算最大回撤
def get_maxDrawBack(port_netVal):
    # hist_high = np.maximum.accumulate(port_netVal)
    drawBack = port_netVal/port_netVal.cummax()-1
    # enddate = np.argmax(drawBack)
    maxDrawBack = np.max(-drawBack)
    # startdate = hist_high[hist_high == hist_high[i]].index[0]

    return maxDrawBack


# 第一步：导入所需要数据  皆为月频数据
# 1、选股池
# stockPool = pd.read_csv(inpath + '/中证500成份权重.csv', index_col=[0])
stockPool = pd.read_csv(inpath + '/沪深300成份权重.csv', index_col=[0])
stockPool = change_format(stockPool)
stockPool = stockPool.dropna()

# 全市场选股
# stockPool = stockPool.fillna(0)

# 2.股票预期收益  这里采用过去12月收益均值 作为未来一个月的预期收益
# 此值可根据自身需求调整
# price = pd.read_csv(inpath+'/复权收盘价.csv',index_col=[0])
# price.index = pd.to_datetime(price.index)
# mu = price.pct_change().rolling(12).mean()
# mu = change_format(mu)

mu = -pd.read_csv(inpath + '/RealizedVolatility_60D.csv', index_col=[0])
mu = change_format(mu)

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


# 第二步：提取月频数据
# 1、设定日期并提取当月数据
# date_list = ['2013-12', '2014-01', '2014-02', '2014-03', '2014-04',
#              '2014-05', '2014-06', '2014-07', '2014-08', '2014-09']
# date_list = ['2013-12', '2014-01', '2014-02', '2014-03', '2014-04',
#              '2014-05', '2014-06', '2014-07', '2014-08', '2014-09',
#              '2014-10', '2014-11', '2014-12', '2015-01', '2015-02',
#              '2015-03', '2015-04', '2015-05', '2015-06', '2015-07',
#              '2015-08', '2015-09', '2015-10', '2015-11', '2015-12',
#              '2016-01', '2016-02', '2016-03', '2016-04', '2016-05',
#              '2016-06', '2016-07', '2016-08', '2016-09', '2016-10',
#              '2016-11', '2016-12', '2017-01', '2017-02', '2017-03',
#              '2017-04', '2017-05', '2017-06', '2017-07', '2017-08',
#              '2017-09', '2017-10', '2017-11', '2017-12', '2018-01',
#              '2018-02', '2018-03', '2018-04', '2018-05', '2018-06',
#              '2018-07', '2018-08', '2018-09', '2018-10', '2018-11',
#              '2018-12', '2019-01', '2019-02', '2019-03', '2019-04',
#              '2019-05', '2019-06']
date_list = ['2015-09', '2015-10', '2015-11']



# stock
ret_daily = pd.read_csv('/Users/zhangchi/富国实习/'
                        '组合优化/组合优化数据1118/retdata_daily.csv',
                        index_col=[0]).sort_index(axis=1)
ret_daily.index = pd.to_datetime(ret_daily.index.astype(str))
ret_daily = ret_daily.fillna(0)+1

price_day = pd.read_csv(open(inpath+'/复权收盘价_day.csv'), index_col=[0])
price_day.index = pd.to_datetime(price_day.index)

# benchmark net value
# index
hs_price = pd.read_csv('/Users/zhangchi/富国实习/组合优化/优化器/Input/hs_price.csv', index_col=0)
hs_price.index = pd.to_datetime(hs_price.index.astype(str))
b_netVal = hs_price.loc[date_list[1]:date_list[-1], :]
b_netVal = b_netVal / b_netVal.iloc[0]
# b_netVal.to_csv('/Users/zhangchi/富国实习/组合优化/Output/中证500增强/基准净值曲线.csv', header=True)
# b_netVal.to_csv('/Users/zhangchi/富国实习/组合优化/Output/沪深300增强/基准净值曲线.csv', header=True)
b_ret = b_netVal.iloc[-1].values[0] - 1
# portfolio return
port_ret = pd.Series()

# 每期选出的股票数
stockNum_picked = []
# 每期年化波动率
vol_Y = []


for i in range(len(date_list)-1):

    date = date_list[i]
    next_date = date_list[i+1]

    stock_date_M = stock_data.loc[date].dropna()  # 只保留  收益来源项   市值  行业

    # 2.股票协方差矩阵
    Cov = pd.read_csv(inpath + '/协方差矩阵/cov_' + date + '.csv', index_col=[0])
    code_list = stock_date_M.index.get_level_values(1)
    Cov = Cov[code_list].loc[code_list]

    # 第三步：每月基于均值方差模型求解
    # 1、设定相关参数
    n = len(Cov)  # 资产个数
    mu = stock_date_M['预期收益'].values.tolist()  # 期望收益

    cov = Cov.copy().values.tolist()
    GT = linalg.cholesky(cov)  # 协方差矩阵

    xb = stock_date_M['基准指数权重'].values.tolist()

    size = stock_date_M['流通市值'].values.tolist()  # 市值暴露
    industry = pd.get_dummies(stock_date_M['行业']).T.values.tolist()  # 行业暴露

    gamma = 0.15 / np.sqrt(12)  # 年化波动率阈值  一般控制在2%  5%  8%

    lmbd = 15

    sizeStockLimit = 0.05  # 市值暴露阈值

    industryStockLimit = 0.05  # 行业暴露阈值

    # 只有均值
    # w = BasicMarkowitz(n,mu,GT,xb,size,industry,gamma,sizeStockLimit,industryStockLimit)

    # 均值 - 风险惩罚
    w = BasicMarkowitzWithRiskPenalty(n,mu,GT,xb,size,industry,lmbd,gamma,sizeStockLimit,industryStockLimit)

    w[np.abs(w) < 1e-8] = 0

    # weight = pd.DataFrame()
    # weight['date'] = next_date
    # weight['stcode'] = code_list
    # weight['weight'] = w
    #
    # # weight.to_csv('/Users/zhangchi/富国实习/组合优化/Output/中证500增强/指数内增强权重/'+next_date+'_weight.csv', header=True)
    # # weight.to_csv('/Users/zhangchi/富国实习/组合优化/Output/沪深300增强/指数内增强权重/'+next_date+'_weight.csv', header=True)
    #
    # # 可以封装函数
    # # 存在问题：如果本月第一个交易日就停牌，价格会为空
    # next_M_price = price_day.loc[next_date, code_list].fillna(method='ffill').fillna(1).apply(lambda x:x/x[0])
    #
    # # portfolio net value
    # M_portVal = pd.Series(np.zeros(next_M_price.shape[0]))
    #
    # for j in range(next_M_price.shape[0]):
    #     netVal = np.dot(w, next_M_price.iloc[j, :])
    #     M_portVal.iloc[j] = netVal
    #
    # M_portret = M_portVal.pct_change().fillna(M_portVal[0]-1)
    # port_ret = pd.concat([port_ret, M_portret], axis=0)

    # 3、求解结果展示
    print('========'+date+'优化结果'+'========')
    stockNum_picked.append(len(w[w!=0]))
    print('股票数量为', len(w[w!=0]))
    print('权重求和为', w.sum())
    print('最大权重为', w.max())
    print('最小权重为', w[w!=0].min())
    # print('期望收益为:', round(np.dot(mu, w) * 100, 3), '%')
    # print('组合波动率', np.sqrt(np.dot(np.dot(w.T, cov), w)))
    # print('组合年化波动率', np.sqrt(np.dot(np.dot(w.T, cov), w))*np.sqrt(12))


    # pd.DataFrame(w).to_csv(date+'权重.csv')

    # 主动权重
    # x = w - xb
    # print('市值暴露为', np.dot(size, x))
    #
    # print('行业暴露为\n', np.dot(industry, x))

    # print('跟踪误差', np.sqrt(np.dot(np.dot(x.T, cov), x)))
    #
    #
    # vol = np.sqrt(np.dot(np.dot(x.T, cov), x)) * np.sqrt(12)
    # vol_Y.append(vol)
    # print('年化跟踪误差', vol)

# port_ret.index = b_netVal.index
# # port_ret.to_csv('/Users/zhangchi/富国实习/组合优化/Output/中证500增强/全市场增强组合日度收益.csv', header=True)
# # port_ret.to_csv('/Users/zhangchi/富国实习/组合优化/Output/沪深300增强/全市场增强组合日度收益.csv', header=True)
# port_netVal = (1+port_ret).cumprod()
# # port_netVal = port_netVal / port_netVal[0]
# p_ret = port_netVal.iloc[-1] - 1
#
# maxDrawBack = get_maxDrawBack(port_netVal)
# print('*******************************')
# print('最大回撤：', maxDrawBack)
# # print('最大回撤开始日期：', startdate)
# # print('最大回撤结束日期：', enddate)
# print('*******************************')
# print('基准收益：', b_ret)
# print('策略收益：', p_ret)
print('*******************************')
print('月均股票数量：', np.mean(stockNum_picked))

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(port_netVal, label='portfolio')
# ax.plot(b_netVal, label='benchmark')
# ax.set_xlabel('Stages')
# ax.set_ylabel('Net')
# # ax.set_title('ZZ500')
# ax.set_title('HS300')
# # plt.savefig('/Users/zhangchi/富国实习/组合优化/Output/'
# #             '沪深300增强/图片/沪深300指数内增强净值曲线.png',
# #             dpi=400, bbox_inches='tight')
# ax.legend(loc='best')
#
# fig.show()
