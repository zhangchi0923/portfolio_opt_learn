import pandas as pd
import numpy as np
import datetime as datetime
from dateutil.parser import parse


thispath = '/Users/zhangchi/富国实习/组合优化/组合优化数据1118'

# 预期收益
def get_mu(date):
    mu = pd.read_csv(thispath+'/最终结果/'+date+'.csv', index_col=[0])*0.01
    date_idx = [date]*mu.shape[0]
    mu.index = [pd.to_datetime(date_idx), mu.index]
    mu.columns = ['预期收益']
    return mu

# mu = get_mu('2018-01-31')

# 股票池（沪深300成分内）
def get_stockPool(pool='hs300'):
    if pool == 'hs300':
        stockPool = pd.read_csv(thispath + '/HS300成分权重.csv', index_col=[0])
        stockPool.index = pd.to_datetime(stockPool.index.values)
        index2 = stockPool['代码']
        stockPool.drop('代码', axis=1, inplace=True)
        stockPool.index = [stockPool.index, index2]
        return stockPool


stpool = get_stockPool()

stlist = ['000001.SZ', '000002.SZ', '000004.SZ', '000011.SZ', '000166.SZ']
# 协方差矩阵估计
def get_cov_estimator(stock, date, window=252, method='shrink_id'):

    raw_ret = pd.read_csv(thispath + '/retdata_daily.csv', index_col=[0])
    date_idx = raw_ret.index.values.tolist().index(date)
    ret = raw_ret.iloc[date_idx:date_idx + window, :]
    ret = ret.loc[:, stock].fillna(0)  # 空值先不管
    samp_cov = ret.cov().values
    samp_corr = ret.corr().values
    ret = ret.values        # 转为 ndarray
    stock_num = ret.shape[1]

    if method == 'shrink_id':
        # mu estimator
        # mu_estimator = np.trace(samp_cov)/stock_num
        mu_estimator = np.linalg.norm(samp_cov, 'fro')
        shrink_target = mu_estimator * np.eye(stock_num)

        # delta estimator
        delta_estimator = np.linalg.norm(samp_cov - shrink_target, 'fro')

        # beta estimator
        beta_serie = []
        for i in range(window):
            prod = np.mat(ret[i]).T*np.mat(ret[i])
            beta_serie.append(np.linalg.norm(prod - samp_cov, 'fro'))
        beta_estimator_ = np.mean(beta_serie)/(window-1)
        beta_estimator = np.minimum(delta_estimator, beta_estimator_)

        # shrink intensity
        beta = beta_estimator / delta_estimator

        # cov estimator
        cov_estimator = beta*shrink_target + (1-beta)*samp_cov

    elif method == 'const_corr':
        corr_num = stock_num * (stock_num - 1) / 2
        avg_corr = np.nansum(np.tril(samp_corr, -1)) / corr_num  # nan设置为0
        ## 构建压缩目标

        ## 固定相关系数法
        stock_var = np.diag(samp_cov)  # 提取各股票方差
        stock_std = np.sqrt(stock_var)
        shrink_target = np.diag(stock_var)  # 设置对角元

        tri_up = np.zeros([stock_num, stock_num])
        for i in range(samp_cov.shape[0]):
            for j in range(i + 1, stock_num):
                tri_up[i, j] = stock_std[i] * stock_std[j] * avg_corr

        shrink_target += tri_up + tri_up.T

        ## pi consistent estimator (synmetric)
        avg_ret = np.mean(ret, axis=0)  # average return of each stock
        pi_mat = np.zeros([stock_num, stock_num])
        for i in range(stock_num):
            for j in range(i, stock_num):
                pi_serie = []
                for k in range(window):
                    pi_serie.append((ret[k][i] - avg_ret[i]) * (ret[k][j] - avg_ret[j]) - samp_cov[i][j])
                pi_serie = np.power(pi_serie, 2)
                pi_mat[i][j] = np.sum(pi_serie)
        pi_mat += np.triu(pi_mat, 1).T

        pi_estimator = np.sum(pi_mat)

        ## rho consistent estimator (synmetric)
        rho_mat = np.zeros([stock_num, stock_num])
        for i in range(stock_num):
            for j in range(i + 1, stock_num):
                rho_serie_i = []
                rho_serie_j = []
                for k in range(window):
                    multi = (ret[k][i] - avg_ret[i]) * (ret[k][j] - avg_ret[j]) - samp_cov[i][j]
                    multi_i = np.power(ret[k][i], 2) - samp_cov[i][i]
                    multi_j = np.power(ret[k][j], 2) - samp_cov[j][j]

                    rho_serie_i.append(multi * multi_i)
                    rho_serie_j.append(multi * multi_j)

                theta_ii_ij = np.mean(rho_serie_i)
                theta_jj_ij = np.mean(rho_serie_j)
                rho_mat[i][j] = 0.5 * avg_corr * (np.sqrt(samp_cov[j][j] / samp_cov[i][i]) * theta_ii_ij
                                                  + np.sqrt(samp_cov[i][i] / samp_cov[j][j]) * theta_jj_ij)

        for i in range(rho_mat.shape[0]):
            rho_mat[i][i] = pi_mat[i][i]

        rho_mat += np.triu(rho_mat, 1).T
        rho_estimator = np.sum(rho_mat)

        ## gamma consistent estimator
        gamma_estimator = np.linalg.norm((shrink_target - samp_cov), 'fro')

        ## kappa consistent estimator
        kappa_estimator = (pi_estimator - rho_estimator) / gamma_estimator

        ## obtain optimal shrinkage intensity
        beta = np.maximum(0, np.minimum(kappa_estimator / window, 1))
        cov_estimator = beta*shrink_target + (1-beta)*samp_cov

    elif method == 'rand_mat':

        # generate random corr matrix
        rand_mat = np.random.rand(window, stock_num)
        rand_cov = np.cov(rand_mat, rowvar=False)
        diag = np.sqrt(np.diag(rand_cov))

        rand_corr = np.corrcoef(rand_mat, rowvar=False)

        rand_eig = np.linalg.eigvals(rand_corr)
        # obtain max eigenvalue
        rand_max_eig = rand_eig.max()

        # obtain samp eigenvalue and its eigenvector
        samp_eig, samp_eigvec = np.linalg.eig(samp_corr)
        idx = np.argwhere(samp_eig > rand_max_eig)
        info_mat = np.zeros_like(samp_corr)
        for i in range(idx.shape[0]):
            weizhi = idx[i][0]
            info_mat += samp_eig[i] * np.mat(samp_eigvec[:, weizhi]).T*np.mat(samp_eigvec[:, weizhi])

            samp_eig[idx[i][0]] = 0

        alpha = samp_eig.mean()

        corr_estimator = info_mat+alpha*np.eye(stock_num)
        cov_estimator = diag*corr_estimator*diag

    eig = np.linalg.eigvals(cov_estimator)
    if np.all(eig) > 0:
        return cov_estimator
    else:
        print('Non invertible Covariance!')
        return None


# cov = get_cov_estimator(stlist, '2018-01-31', window=21, method='rand_mat')