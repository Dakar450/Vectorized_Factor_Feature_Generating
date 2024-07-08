# -*- coding: UTF-8 -*-
import numpy as np
#import pandas as pd


class Momentum:
    def __init__(self):
        self.funcs = {
            'MA': lambda x, loc, period: np.mean(x[:, loc, 0:period], axis=1), # Moving average
            'EMA': lambda x, loc, period: self.ema(x, loc, period), # exponential moving average
            'max': lambda x, loc, period: np.max(x[:, loc, 0:period], axis=1), # period max of return
            'max_mul': lambda x, loc, period, num: np.mean(np.sort(x[:, loc, 0:period], axis=1)[:,period-num:], axis=1), # 最大num个的均值
            'dif_MA': lambda x, loc, period: np.mean(x[:, loc, 0:period][:, 0::2]-x[:, loc, 0:period][:, 1::2], axis=1),
            'MACD_dif': lambda x, loc, periodshort, periodlong: self.ema(x, loc, periodshort)-self.ema(x, loc, periodlong),
            'max_up': lambda x, loc, period: self.max_up_all(x[:, loc, 0:period]), # 最大上扬
            'H_O_max': lambda x, loc1, loc2, period: np.max((x[:, loc1, 0:period]-x[:, loc2, 0:period])/x[:, loc2, 0:period], axis=1), # period max of (high-open)/open
            'H_O_max_mul': lambda x, loc1, loc2, period, num: np.mean(np.sort((x[:, loc1, 0:period]-x[:, loc2, 0:period])/x[:, loc2, 0:period],axis=1)[:, period-num:], axis=1) # period mean of num max of (high-open)/open
        }

    def ema(self, data, loc, period):
        ema = data[:, loc, period-1]
        alpha = 2 / (period + 1)
        for t in range(1, period):
            ema = (alpha * data[:, loc, period-t-1] + (1 - alpha) * ema)
        return ema

    def max_up_all(self, wealth_tensor):
        wealth_tensor = np.flip(wealth_tensor, axis=1)
        def max_up(wealth): # 动态规划计算最大上扬
            maxup_so_far, max_local, min_local = 0, wealth[0], wealth[0]
            for wealth_i in wealth:
                if wealth_i <= min_local:
                    if maxup_so_far < (max_local - min_local) / min_local:
                        maxup_so_far = (max_local - min_local) / min_local
                    max_local = wealth_i
                    min_local = wealth_i
                elif wealth_i > max_local:
                    max_local = wealth_i
            if maxup_so_far == 0:
                maxup_so_far = (max_local - min_local) / min_local
            return maxup_so_far
        maxup = []
        for data_i in wealth_tensor:
            maxup.append(max_up(data_i))
        return np.array(maxup)


class HighMoment:
    def __init__(self):
        self.funcs = {
            'high_moment': lambda x, loc, period, nmoment: np.mean((x[:,loc,0:period]-np.mean(x[:,loc,0:period], axis=1)[:, np.newaxis])**nmoment, axis=1), # 高阶矩
            'scale_high_moment': lambda x, loc, period, nmoment: np.mean((x[:,loc,0:period]-np.mean(x[:,loc,0:period], axis=1)[:, np.newaxis])**nmoment, axis=1)/np.std(x[:, loc, 0:period], axis=1)**nmoment,
            'cohigh_mom': lambda x, loci, locm, period, nmoment_i, nmoment_m: self.cohigh_moment(x, loci, locm, period, nmoment_i, nmoment_m) # co-高阶矩，其中nomoment_i是该币收益率的矩，nmoment_m是市场收益率的矩
        }
    def cohigh_moment(self, x, loci, locm, period, nmoment_i, nmoment_m):
        ri_0 = x[:, loci, 0:period]-np.mean(x[:, loci, 0:period], axis=1)[:, np.newaxis]
        rm_0 = x[:, locm, 0:period]-np.mean(x[:, locm, 0:period], axis=1)[:, np.newaxis]
        cohigh = np.mean(ri_0**nmoment_i*rm_0**nmoment_m, axis=1)
        coscale = np.sqrt(np.mean(ri_0**2, axis=1)**nmoment_i*np.mean(rm_0**2, axis=1)**nmoment_m)
        return cohigh/coscale

class Liquidity:
    def __init__(self):
        self.funcs = {
            'MA': lambda x, loc, period: np.mean(x[:, loc, 0:period], axis=1),
            'std': lambda x, loc, period: np.std(x[:, loc, 0:period], axis=1),
            'each_MA': lambda x, loc_qv, loc_co, period: np.mean(x[:, loc_qv, 0:period]/x[:, loc_co, 0:period], axis=1),
            'each_std': lambda x, loc_qv, loc_co, period: np.std(x[:, loc_qv, 0:period]/x[:, loc_co, 0:period], axis=1),
            'per_MA': lambda x, loc_r, loc_qv, period: np.mean(x[:, loc_r, 0:period]/x[:, loc_qv, 0:period], axis=1),
            'abs_per_MA': lambda x, loc_r, loc_qv, period: abs(np.mean(x[:, loc_r, 0:period]/x[:, loc_qv, 0:period], axis=1)),
        }


class Volatility:
    def __init__(self):
        self.funcs = {
            'std': lambda x, loc, period: np.std(x[:, loc, 0:period], axis=1),
            'min': lambda x, loc, period: np.min(x[:, loc, 0:period], axis=1),
            'min_mul': lambda x, loc, period, num: np.mean(np.sort(x[:, loc, 0:period], axis=1)[:,0:num], axis=1),
            'max_drawdown': lambda x, loc, period: self.maxdraw_all(x[:, loc, 0:period]),
            'Bol_Position': lambda x, loc, period: (x[:, loc, 0]-np.mean(x[:, loc, 0:period], axis=1))/np.std(x[:, loc, 0:period], axis=1),
            'abs_Bol_Position': lambda x, loc, period: abs((x[:, loc, 0]-np.mean(x[:, loc, 0:period], axis=1))/np.std(x[:, loc, 0:period], axis=1)),
            'LO_min': lambda x, locO, locmin, period: np.min((x[:, locmin, 0:period]-x[:, locO, 0:period])/x[:, locO, 0:period], axis=1),
            'LO_MA': lambda x, locO, locmin, period: np.mean((x[:, locmin, 0:period]-x[:, locO, 0:period])/x[:, locO, 0:period], axis=1),
            'LO_min_mul': lambda x, loc1, loc2, period, num: np.mean(np.sort((x[:, loc1, 0:period]-x[:, loc2, 0:period])/x[:, loc2, 0:period],axis=1)[:, 0:num], axis=1),
            'OHLC_var': lambda x, locO, locH, locL, locC, period: np.mean(1/2*np.log(x[:, locH, 0:period]/x[:, locL, 0:period])**2-(2*np.log(2)-1)*np.log(x[:, locC, 0:period]/x[:, locO, 0:period])**2, axis=1),
            'low_var': lambda x, loc, threshold, period: self.low_var(x[:,loc,0:period], threshold),
            'up_beta':lambda x, loci, locm, period, threshold:self.half_beta(x, loci, locm, threshold, period, True), # cov(ri,rm)/var(rm)|rm>=threshold
            'down_beta':lambda x, loci, locm, period, threshold:self.half_beta(x, loci, locm, threshold, period, False), # cov(ri,rm)/var(rm)|rm<=threshold
            'H_TCR': lambda x, loci, locm, threshold, period: self.hybrid_tail_risk(x, loci, locm, threshold, period) # hybrid_tail_risk, cov(ri,rm|ri<=threshold)
        }

    def maxdraw_all(self, wealth_tensor):
        wealth_tensor = np.flip(wealth_tensor, axis=1)
        def max_drawdown(wealth): # dynamic programming calculate max drawdown
            maxdraw_so_far, max_local, min_local = 0, wealth[0], wealth[0]
            for wealth_i in wealth:
                if wealth_i >= max_local:
                    if maxdraw_so_far < (max_local - min_local) / max_local:
                        maxdraw_so_far = (max_local - min_local) / max_local
                    max_local = wealth_i
                    min_local = wealth_i
                elif wealth_i < min_local:
                    min_local = wealth_i
            if maxdraw_so_far == 0:
                maxdraw_so_far = (max_local - min_local) / max_local
            return maxdraw_so_far
        max_draw = []
        for data_i in wealth_tensor:
            max_draw.append(max_drawdown(data_i))
        return np.array(max_draw)

    def low_var(self, x, threshold):
        quantile1 = np.quantile(x, threshold, axis=1)
        low_square = np.mean((x <= quantile1[:, np.newaxis]).astype(int)*x**2, axis=1)
        return low_square/threshold

    def hybrid_tail_risk(self, data, loci, locm, threshold, period):
        xm = np.quantile(data[:, locm, 0:period], threshold, axis=1)
        xi = np.quantile(data[:, loci, 0:period], threshold, axis=1)
        condition = (data[:, loci, 0:period]<=xi[:, np.newaxis]).astype(int)
        xi0 = data[:, loci, 0:period]-xi[:, np.newaxis]
        xm0 = data[:, locm, 0:period]-xm[:, np.newaxis]
        H_TCR = np.mean(xi0*xm0*condition, axis=1)/threshold
        return H_TCR

    def half_beta(self, data, loci, locm, threshold, period, up_bool):
        xm = np.quantile(data[:, locm, 0:period], threshold, axis=1)
        if up_bool == True:
            condition = (data[:, locm, 0:period]>=xm[:, np.newaxis]).astype(int)
            scale = 1/(1-threshold)
        else:
            condition = (data[:, locm, 0:period]<=xm[:, np.newaxis]).astype(int)
            scale = 1/threshold
        xm1 = data[:, locm, 0:period]
        xi1 = data[:, loci, 0:period]
        xm0 = condition*(xm1-scale*np.mean(xm1*condition,axis=1)[:, np.newaxis])
        xi0 = condition*(xi1-scale*np.mean(xi1*condition, axis=1)[:,np.newaxis])
        return np.sum(xi0*xm0, axis=1)/np.sum(xm0*xm0, axis=1)

class Correlation:
    def __init__(self):
        self.funcs = {
            'cor': lambda x, locp, locv, period: self.corfunc(x, locp, locv, period),
            'up_cor': lambda x, locp, locv, locCon, period: self.halfcorfunc(x, locp, locv, locCon, period, True),
            'down_cor': lambda x, locp, locv, locCon, period: self.halfcorfunc(x, locp, locv, locCon, period, False),
        }

    def corfunc(self, x, locp, locv, period):
        p = x[:, locp, 0:period]
        v = x[:, locv, 0:period]
        p0 = p-np.mean(p, axis=1)[:, np.newaxis]
        v0 = v-np.mean(v, axis=1)[:, np.newaxis]
        return np.mean(p0*v0, axis=1)/(np.std(p, axis=1)*np.std(v, axis=1))

    def halfcorfunc(self, x, locp, locv, locCon, period, upbool):
        p = x[:, locp, 0:period]
        v = x[:, locv, 0:period]
        xCon = np.quantile(x[:, locCon, 0:period], 0.5, axis=1)
        if upbool == True:
            condition = (x[:, locCon, 0:period] >= xCon[:, np.newaxis]).astype(int)
        else:
            condition = (x[:, locCon, 0:period] <= xCon[:, np.newaxis]).astype(int)
        p0 = condition * (p - 2 * np.mean(p * condition, axis=1)[:, np.newaxis])
        v0 = condition * (v - 2 * np.mean(v * condition, axis=1)[:, np.newaxis])
        return np.mean(p0 * v0, axis=1) / np.sqrt(np.mean(p0 * p0, axis=1)*np.mean(v0 * v0, axis=1))

