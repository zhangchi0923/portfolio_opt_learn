import matplotlib.pyplot as plt
import pandas as pd

path = '/Users/zhangchi/富国实习/组合优化'

# # 中证500指数内增强，与基准对比
# port_ret = pd.read_csv(path+'/Output/中证500增强/指数内增强组合日度收益.csv', index_col=[0])
# port_ret.index = pd.to_datetime(port_ret.index.astype(str))
# port_val = (1+port_ret).cumprod()
#
# b_val = pd.read_csv(path+'/优化器/Input/zz_price.csv', index_col=[0])
# b_val.index = pd.to_datetime(b_val.index.astype(str))
# b_val = b_val.loc['2014-01':'2019-06', :].apply(lambda x: x/x[0])
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.plot(port_val, linestyle='-', color='#B22222', label='portfolio')
# ax.plot(b_val, linestyle='-', color='#4682B4', label='benchmark')
# ax.set_xlim(['1/1/2014', '1/5/2019'])
# ax.legend(loc='best', fontsize='large')
# plt.savefig(path+'/Output/中证500增强/图片/中证500指数内增强净值曲线.png',
#             dpi=400, bbox='tight')
# fig.show()

# # 沪深300指数内增强，与基准对比
# port_ret = pd.read_csv(path+'/Output/沪深300增强/指数内增强组合日度收益.csv', index_col=[0])
# port_ret.index = pd.to_datetime(port_ret.index.astype(str))
# port_val = (1+port_ret).cumprod()
#
# b_val = pd.read_csv(path+'/优化器/Input/hs_price.csv',
#                     index_col=[0])
# b_val = b_val.loc['2014-01-01':'2019-06', :].apply(lambda x: x/x[0])
# b_val.index = pd.to_datetime(b_val.index.astype(str))
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.plot(port_val, linestyle='-', color='#B22222', label='portfolio')
# ax.plot(b_val, linestyle='-', color='#4682B4', label='benchmark')
# ax.set_xlim(['1/1/2014', '1/5/2019'])
# ax.legend(loc='best', fontsize='large')
# plt.savefig(path+'/Output/沪深300增强/图片/沪深300指数内增强净值曲线.png',
#             dpi=400, bbox='tight')
# fig.show()

# 均值模型与均值方差模型对比
port_ret_mo = pd.read_csv(path+'/Output/沪深300增强/指数内增强组合日度收益.csv', index_col=[0])
port_ret_mvo = pd.read_csv(path+'/Output/沪深300增强/指数内增强组合日度收益（均值模型）.csv', index_col=[0])
port_ret_mo.index = pd.to_datetime(port_ret_mo.index.astype(str))
port_ret_mvo.index = pd.to_datetime(port_ret_mvo.index.astype(str))

port_val_mo = (1+port_ret_mo).cumprod()
port_val_mvo = (1+port_ret_mvo).cumprod()

b_val = pd.read_csv(path+'/优化器/Input/hs_price.csv', index_col=[0])
b_val.index = pd.to_datetime(b_val.index.astype(str))
b_val = b_val.loc['2014-01':'2019-06', :].apply(lambda x: x/x[0])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(port_val_mo, linestyle='-', color='#FF6347',
        label='Mean Opt')
ax.plot(port_val_mvo, linestyle='-', color='#B22222',
        label='Mean Var Opt')
ax.plot(b_val, linestyle=':', color='#4682B4',
        label='Benchmark')
ax.set_xlim(['1/1/2014', '1/1/2020'])
# ax.set_ylim([-0.5, 1])
ax.legend(loc='best', fontsize='large')
plt.savefig(path+'/Output/沪深300增强/图片/均值模型vs均值方差模型.png',
             dpi=400, bbox='tight')
fig.show()

# # 不同风险厌恶系数对组合收益的影响
# port_5 = pd.read_csv(path+'/Output/沪深300增强/lambda=5指数内增强组合日度收益.csv', index_col=[0])
# port_10 = pd.read_csv(path+'/Output/沪深300增强/lambda=10指数内增强组合日度收益.csv', index_col=[0])
# port_15 = pd.read_csv(path+'/Output/沪深300增强/lambda=15指数内增强组合日度收益.csv', index_col=[0])
# port_20 = pd.read_csv(path+'/Output/沪深300增强/lambda=20指数内增强组合日度收益.csv', index_col=[0])
# port_25 = pd.read_csv(path+'/Output/沪深300增强/lambda=25指数内增强组合日度收益.csv', index_col=[0])
#
# port_5.index = pd.to_datetime(port_5.index.astype(str))
# port_10.index = pd.to_datetime(port_10.index.astype(str))
# port_15.index = pd.to_datetime(port_15.index.astype(str))
# port_20.index = pd.to_datetime(port_20.index.astype(str))
# port_25.index = pd.to_datetime(port_25.index.astype(str))
#
# port_5 = (1+port_5).cumprod()
# port_10 = (1+port_10).cumprod()
# port_15 = (1+port_15).cumprod()
# port_20 = (1+port_20).cumprod()
# port_25 = (1+port_25).cumprod()
#
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.plot(port_5, linestyle='-',
#         label='lambda=5')
# ax.plot(port_10, linestyle='-',
#         label='lambda=10')
# ax.plot(port_15, linestyle='-',
#         label='lambda=15')
# ax.plot(port_20, linestyle='-',
#         label='lambda=20')
# ax.plot(port_25, linestyle='-',
#         label='lambda=25')
#
# b_val = pd.read_csv(path+'/优化器/Input/hs_price.csv',
#                     index_col=[0]).apply(lambda x: x / x[0])
# b_val.index = pd.to_datetime(b_val.index.astype(str))
# ax.set_xlim(['1/1/2014', '1/5/2019'])
# # ax.set_ylim([-0.5, 1])
# ax.legend(loc='best', fontsize='large')
# plt.savefig(path+'/Output/沪深300增强/图片/不同风险厌恶系数.png',
#              dpi=400, bbox='tight')
# fig.show()



# # 跟踪误差对组合收益的影响
# port_5 = pd.read_csv(path+'/Output/沪深300增强/gamma=0.05指数内增强组合日度收益.csv', index_col=[0])
# port_10 = pd.read_csv(path+'/Output/沪深300增强/gamma=0.08指数内增强组合日度收益.csv', index_col=[0])
# port_15 = pd.read_csv(path+'/Output/沪深300增强/gamma=0.1指数内增强组合日度收益.csv', index_col=[0])
# port_20 = pd.read_csv(path+'/Output/沪深300增强/gamma=0.2指数内增强组合日度收益.csv', index_col=[0])
#
# port_5.index = pd.to_datetime(port_5.index.astype(str))
# port_10.index = pd.to_datetime(port_10.index.astype(str))
# port_15.index = pd.to_datetime(port_15.index.astype(str))
# port_20.index = pd.to_datetime(port_20.index.astype(str))
#
# port_5 = (1+port_5).cumprod()
# port_10 = (1+port_10).cumprod()
# port_15 = (1+port_15).cumprod()
# port_20 = (1+port_20).cumprod()
#
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.plot(port_5, linestyle='-',
#         label='gamma=0.05')
# ax.plot(port_10, linestyle='-',
#         label='gamma=0.08')
# ax.plot(port_15, linestyle='-',
#         label='gamma=0.1')
# ax.plot(port_val, linestyle='-',
#         label='gamma=0.15')
# ax.plot(port_20, linestyle='-',
#         label='gamma=0.2')
#
# b_val = pd.read_csv(path+'/优化器/Input/hs_price.csv',
#                     index_col=[0]).apply(lambda x: x / x[0])
# b_val.index = pd.to_datetime(b_val.index.astype(str))
# # ax.plot(b_val, linestyle=':',
# #         label='benchmark')
#
# ax.set_xlim(['1/1/2014', '1/5/2019'])
# # ax.set_ylim([-0.5, 1])
# ax.legend(loc='best', fontsize='large')
# plt.savefig(path+'/Output/沪深300增强/图片/不同目标跟踪误差.png',
#              dpi=400, bbox='tight')
# fig.show()

# # 中证500全市场增强，与基准对比
# port_ret = pd.read_csv(path+'/Output/中证500增强/全市场增强组合日度收益.csv', index_col=[0])
# port_ret.index = pd.to_datetime(port_ret.index.astype(str))
# port_val = (1+port_ret).cumprod()
#
# port_ret_ = pd.read_csv(path+'/Output/中证500增强/指数内增强组合日度收益.csv', index_col=[0])
# port_ret_.index = pd.to_datetime(port_ret.index.astype(str))
# port_val_ = (1+port_ret_).cumprod()
#
#
# b_val = pd.read_csv(path+'/优化器/Input/zz_price.csv', index_col=[0])
# b_val.index = pd.to_datetime(b_val.index.astype(str))
# b_val = b_val.loc['2014-01':'2019-06', :].apply(lambda x: x/x[0])
#
# active_val1 = port_val.iloc[:,0] - b_val['close']
# active_val2 = port_val_.iloc[:,0] - b_val['close']
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# # ax.plot(port_val, linestyle='-', color='#4682B4', label='all')
# ax.plot(port_val, linestyle='-', color='#B22222', label='portfolio')
# # ax.plot(port_val_, linestyle='-', color='#B0C4DE', label='inside')
# ax.plot(b_val, linestyle='-', color='#4682B4', label='benchmark')
# # ax.plot(active_val1, linestyle='-', color='#4682B4', label='all')
# # ax.plot(active_val2, linestyle='-', color='#B0C4DE', label='inside')
# ax.set_xlim(['1/1/2014', '1/5/2019'])
# ax.legend(loc='best', fontsize='large')
# plt.savefig(path+'/Output/中证500增强/图片/中证500全市场增强净值曲线.png',
#             dpi=400, bbox='tight')
# fig.show()


# # 沪深300全市场增强，与基准对比
# port_ret = pd.read_csv(path+'/Output/沪深300增强/全市场增强组合日度收益.csv', index_col=[0])
# port_ret.index = pd.to_datetime(port_ret.index.astype(str))
# port_val = (1+port_ret).cumprod()
#
# port_ret_ = pd.read_csv(path+'/Output/沪深300增强/指数内增强组合日度收益.csv', index_col=[0])
# port_ret_.index = pd.to_datetime(port_ret.index.astype(str))
# port_val_ = (1+port_ret_).cumprod()
#
#
# b_val = pd.read_csv(path+'/优化器/Input/hs_price.csv', index_col=[0])
# b_val.index = pd.to_datetime(b_val.index.astype(str))
# b_val = b_val.loc['2014-01':'2019-06', :].apply(lambda x: x/x[0])
#
# active_val1 = port_val.iloc[:,0] - b_val['close']
# active_val2 = port_val_.iloc[:,0] - b_val['close']
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# # ax.plot(port_val, linestyle='-', color='#4682B4', label='all')
# # ax.plot(port_val, linestyle='-', color='#B22222', label='portfolio')
# # ax.plot(port_val_, linestyle='-', color='#B0C4DE', label='inside')
# # ax.plot(b_val, linestyle='-', color='#4682B4', label='benchmark')
# # ax.plot(active_val1, linestyle='-', color='#4682B4', label='all')
# ax.plot(active_val2, linestyle='-', color='#B0C4DE', label='inside')
# ax.set_xlim(['1/1/2014', '1/10/2019'])
# ax.legend(loc='best', fontsize='large')
# # plt.savefig(path+'/Output/沪深300增强/图片/沪深300全市场增强净值曲线.png',
# #             dpi=400, bbox='tight')
# fig.show()


# # 沪深300全市场增强，与基准对比
# port_ret = pd.read_csv(path+'/Output/沪深300增强/全市场增强组合日度收益.csv', index_col=[0])
# port_ret.index = pd.to_datetime(port_ret.index.astype(str))
# port_val = (1+port_ret).cumprod()
#
# port_ret_ = pd.read_csv(path+'/Output/沪深300增强/指数内增强组合日度收益.csv', index_col=[0])
# port_ret_.index = pd.to_datetime(port_ret.index.astype(str))
# port_val_ = (1+port_ret_).cumprod()
#
#
# b_val = pd.read_csv(path+'/优化器/Input/hs_price.csv',
#                     index_col=[0]).apply(lambda x: x / x[0])
# b_val.index = pd.to_datetime(b_val.index.astype(str))
# b_val = b_val.loc['2014-01-02':'2019-05-31', :]
#
# active_val1 = port_val.iloc[:, 0] - b_val['close']
# active_val2 = port_val_.iloc[:, 0] - b_val['close']
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(port_val, linestyle='-', color='#B22222', label='portfolio')
# # ax.plot(port_val_, linestyle='-', color='#B0C4DE', label='inside')
# ax.plot(b_val, linestyle='-', color='#4682B4', label='benchmark')
# # ax.plot(active_val1, linestyle='-', color='#4682B4', label='all')
# # ax.plot(active_val2, linestyle='-', color='#B0C4DE', label='inside')
# ax.set_xlim(['1/1/2014', '1/5/2019'])
# ax.legend(loc='best', fontsize='large')
# # plt.savefig(path+'/Output/沪深300增强/图片/沪深300全市场增强净值曲线.png',
# #             dpi=400, bbox='tight')
# fig.show()
