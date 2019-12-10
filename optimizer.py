# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 20:16:16 2019

@author: 瓶子

基于均值方差模型求解权重
"""

import numpy as np
import pandas as pd
from mosek.fusion import *
from scipy import linalg


#设置路径
path = '/Users/zhangchi/富国实习/组合优化/优化器'
inpath = path+'/Input'
outpath = path+'/Output'


#功能函数：改变数据格式，换成双索引
def change_format(data):
    data.index = pd.to_datetime(data.index.astype(str))
    data = np.transpose(data).unstack()     # df -> series
    return data

''''
剔除下列股票
1、不在市
2、上市天数小于180
3、特殊处理
4、涨停
5、复权收盘价为空
'''

def filter_data(data):
    price = pd.read_csv(open(path+'/Input/复权收盘价.csv'),index_col=[0])
    price = change_format(price).sort_index().to_frame('月末收盘价')
   
    state = pd.read_csv(open(path+'/Input/是否在市.csv'),index_col=[0])
    state = change_format(state).sort_index().to_frame('state')
 
    listlen = pd.read_csv(open(path+'/Input/上市天数.csv'),index_col=[0])
    listlen = change_format(listlen).to_frame('上市天数')
    
    ST = pd.read_csv(open(path+'/Input/特殊处理.csv'),index_col=[0])
    ST = change_format(ST).to_frame('特殊处理').fillna(1)
    
    zhandie = pd.read_csv(open(path+'/Input/涨跌停.csv'),index_col=[0])
    zhandie = change_format(zhandie).to_frame('涨跌停')
    
    data = pd.concat([data, price, state, listlen, ST, zhandie], join_axes=[data.index],axis=1)
    
    data = data.query('state == 1 ') 
    data = data.query('上市天数 >= 180 ') 
    data = data.query('特殊处理 == 1 ') 
    data = data.query('涨跌停!= 1 ') 
    data = data.dropna()
    data.index.names = ['date','stcode']  
    return data



''''
优化器   基于均值方差模型  在风险一定的条件下最大化收益
输入参数：n,mu,GT,size,industry,gamma,sizeStockLimit,industryStockLimit
n：资产个数
mu：期望收益率
GT：协方差矩阵的cholesky，为一个上三角矩阵
size：市值因子
industry：行业
gamma：风险阈值
sizeStockLimit：市值暴露阈值
industryStockLimit：行业暴露阈值
'''
def BasicMarkowitz(n,mu,GT,xb,size,industry,gamma,sizeStockLimit,industryStockLimit):

    #设定问题
    with  Model("Basic Markowitz") as M:

        # 设定求解对象，n为维度，同时每个x大于0 
        x = M.variable("x", n)
        
        # 目标：最大化预期主动收益
        M.objective('obj', ObjectiveSense.Maximize, Expr.dot(mu, Expr.sub(x, xb)))
        
        # 约束：风险小于gamma
        M.constraint('risk', Expr.vstack(gamma,Expr.mul(GT, Expr.sub(x,xb))), Domain.inQCone())

        #交易限制1： 权重求和为1
        M.constraint('budget', Expr.sum(x), Domain.equalsTo(1.0))
        
        #交易限制2：每个资产的权重在0，0.1之间
        M.constraint('buy1', x, Domain.lessThan(0.1))
        M.constraint('buy2', x, Domain.greaterThan(0.0))
        
        
        #风格限制1：市值暴露小于 sizeStockLimit
        M.constraint('size1',Expr.dot(size,  Expr.sub(x,xb)  ),Domain.lessThan(sizeStockLimit))
        M.constraint('size2',Expr.dot(size, Expr.sub(x,xb)    ) ,Domain.greaterThan(-sizeStockLimit))
          
          
        #风格限制2：行业暴露小于 industryStockLimit
        M.constraint('industry1',Expr.mul(industry, Expr.sub(x,xb)   ),Domain.lessThan(  industryStockLimit  ))
        M.constraint('industry2',Expr.mul(industry, Expr.sub(x,xb)    ),Domain.greaterThan(-industryStockLimit))
          
        # Solves the model.
        M.solve()
    
        return x.level()


#############################

def BasicMarkowitzWithRiskPenalty(n,mu,GT,xb,size,industry,lmbd,
                                  gamma,sizeStockLimit,industryStockLimit):

    with Model('Basic Markowitz With RiskPenalty') as M:

        x = M.variable('x', n, Domain.greaterThan(0.0))
        s = M.variable('s', 1, Domain.unbounded())

        # 最大化预期主动收益-风险惩罚项
        mudotx = Expr.dot(mu, Expr.sub(x, xb))
        M.objective('obj', ObjectiveSense.Maximize, Expr.sub(mudotx, Expr.mul(lmbd, s)))

        # 约束：风险小于gamma
        M.constraint('risk', Expr.vstack(gamma,Expr.mul(GT, Expr.sub(x,xb)   )), Domain.inQCone())

        M.constraint('variance',
                     Expr.vstack(s, 0.5, Expr.mul(GT, x)), Domain.inRotatedQCone())
        M.constraint('Fully Invested', Expr.sum(x), Domain.equalsTo(1.0))

        # 交易限制2：每个资产的权重在0，0.1之间
        M.constraint('buy1', x, Domain.lessThan(0.1))
        M.constraint('buy2', x, Domain.greaterThan(0.0))

        # 风格限制1：市值暴露小于 sizeStockLimit
        M.constraint('size1', Expr.dot(size, Expr.sub(x, xb)), Domain.lessThan(sizeStockLimit))
        M.constraint('size2', Expr.dot(size, Expr.sub(x, xb)), Domain.greaterThan(-sizeStockLimit))

        # 风格限制2：行业暴露小于 industryStockLimit
        M.constraint('industry1', Expr.mul(industry, Expr.sub(x, xb)), Domain.lessThan(industryStockLimit))
        M.constraint('industry2', Expr.mul(industry, Expr.sub(x, xb)), Domain.greaterThan(-industryStockLimit))

        # Solves the model.
        M.solve()

        return x.level()
