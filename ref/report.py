import numpy as np
import pandas as pd


path = '/Users/zhangchi/富国实习/组合优化'

#测试辅助函数
def report(ret, b_ret, freqcy=1):
    # ret.fillna(0,inplace=True)
    net = (1+ret).cumprod()
    b_net = (1+b_ret).cumprod()
    active_ret = ret['0'] - b_ret['close']

    win_day = active_ret[active_ret>=0].count()/len(active_ret)

    yld_rate = (net.iloc[-1]/net.iloc[0]) ** (252/freqcy/len(net)) - 1 -0 # 年化收益率 无风险收益率为0
    sigma = np.std(ret) * np.sqrt(252/freqcy) # 年化波动率
    real_TE = np.std(active_ret) * np.sqrt(252/freqcy) # 实际跟踪误差

    b_drawdown = b_net/b_net.cummax() - 1
    b_maxdrawdown = max(-b_drawdown['close'])
    drawdown = net/net.cummax() - 1 #回撤
    maxdrawdown = max(-drawdown['0'])
    yld_down_ratio = yld_rate / maxdrawdown    # 收益回撤比
    sharpe = yld_rate / sigma # 夏普比 sharpe ratio

    print('策略净值：', round(net.iloc[-1].values[0], 2))
    print('策略日胜率：', str(round(win_day*100, 2))+'%')
    print('年化收益率：', str(round(yld_rate[0]*100, 2))+'%')
    print('年化波动率：', str(round(sigma[0]*100, 2))+'%')
    print('实际年化跟踪误差：', str(round(real_TE*100, 2))+'%')
    print('Sharpe ratio：', str(round(sharpe[0], 2)))
    print('组合最大回撤：', str(round(maxdrawdown*100, 2))+'%')
    print('基准最大回撤：', str(round(b_maxdrawdown*100, 2))+'%')
    print('收益回撤比：', str(round(yld_down_ratio[0], 2)))
    # summary_table = pd.DataFrame(index=[0])
    # summary_table["年化收益率"] = str(round(yld_rate*100,2))+'%'
    # summary_table["年化波动率"] = str(round(sigma*100,2))+'%'
    # summary_table["Sharpe ratio"] = str(round(sharpe,2))
    # summary_table["最大回撤"] = str(round(maxdrawdown*100,2))+'%'
    # summary_table["收益回撤比"] = str(round(yld_down_ratio,2))
    # summary_table.index = ['回测结果统计']
    # return summary_table

# ret = pd.read_csv(path+'/Output/沪深300增强/指数内增强组合日度收益（均值模型）.csv', index_col=[0])
ret = pd.read_csv(path+'/Output/沪深300增强/指数内增强组合日度收益.csv', index_col=[0])
# ret = pd.read_csv(path+'/Output/中证500增强/全市场增强组合日度收益.csv', index_col=[0])
# ret = pd.read_csv(path+'/Output/沪深300增强/gamma=0.2指数内增强组合日度收益.csv', index_col=[0])
ret.index = pd.to_datetime(ret.index.astype(str))

b_val = pd.read_csv(path+'/优化器/Input/hs_price.csv',index_col=[0])
# b_val = pd.read_csv(path+'/优化器/Input/zz_price.csv',index_col=[0])
b_val.index = pd.to_datetime(b_val.index.astype(str))
b_val = b_val.loc['2013-12-31':'2019-06-30', :]
b_ret = b_val.pct_change()[1:]

report(ret, b_ret)
# net = (1+ret1).cumprod()
# yld_rate = (net.iloc[-1]/net.iloc[0]) ** (252/len(net)) - 1 -0
# sigma = np.std(ret1) * np.sqrt(252)
# drawdown = net/net.cummax() - 1
# maxdrawdown = max(-drawdown['0'])
# yld_down_ratio = yld_rate / maxdrawdown    # 收益回撤比
# sharpe = yld_rate / sigma # 夏普比 sharpe ratio


# net = (1+ret).cumprod()
# b_net = (1+b_ret).cumprod()
# active_ret = ret['0'] - b_ret['close']
#
#
# yld_rate = (net.iloc[-1]/net.iloc[0]) ** (252/len(net)) - 1 -0 # 年化收益率 无风险收益率为0
# sigma = np.std(ret) * np.sqrt(252) # 年化波动率
# real_TE = np.std(active_ret) * np.sqrt(252) # 实际跟踪误差
#
#
# drawdown = net/net.cummax() - 1 #回撤
# maxdrawdown = max(-drawdown['0'])
# b_drawdown = b_net/b_net.cummax() - 1
# b_maxdrawdown = max(-b_drawdown['close'])
# yld_down_ratio = yld_rate / maxdrawdown    # 收益回撤比
# sharpe = yld_rate / sigma # 夏普比 sharpe ratio