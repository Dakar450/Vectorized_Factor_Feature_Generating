# -*- coding: UTF-8 -*-
"""
@Project ：CryptoCS 
@File    ：FeatureBase.py
@Author  ：LeslieChiu
@Date    ：2024/4/11 14:27
"""

import pandas as pd
from tqdm import tqdm
import toml
#from FeatureFunc import Momentum, HighMoment, Volatility, Liquidity, Correlation
from ReturnTool import ReturnTool
from FactorAnalysis import FactorAnalysis
import numpy as np

'''OHLC量价数据按照接入参数逻辑 生成特征面板数据的主接口'''


class FeatureBase:
    def __init__(self):
        self.source_data = None  # ohlc原始数据
        self.time_label = None
        self.code_label = None
        self.frequency = None  # 使用的原始数据频率
        self.feature_type_dict = None # 用于通过key连接feature func的大类接口
        self.drop_raw = None
        self.fillna_media = None
        self.New_threshold = None
        self.markets_data = None
        self.analysis = None
        self.num = None
        self.analysis_start_date = None
        self.vecType = None
        self.period_max = None
        self.beta_period = None

    def create(self, source_data, time_label, code_label, frequency, drop_raw, fillna_media, analysis, analysis_start_date, New_threshold, num, vecType, period_max, beta_period):
        self.source_data = source_data
        self.time_label = time_label
        self.code_label = code_label
        self.frequency = frequency
        self.drop_raw = drop_raw
        self.fillna_media = fillna_media
        self.New_threshold = New_threshold
        if vecType == False:
            rt = ReturnTool().create(self.source_data, self.time_label, self.frequency)
            self.markets_data = rt.market_data()
            from FeatureFunc import Momentum, HighMoment, Volatility, Liquidity, Correlation
            self.feature_type_dict = {
                'Momentum': Momentum,
                'HighMoment': HighMoment,
                'Volatility': Volatility,
                'Liquidity': Liquidity,
                'Correlation': Correlation
            }
        elif vecType == True:
            from vecFeatureFunc import Momentum, HighMoment, Volatility, Liquidity, Correlation
            self.feature_type_dict = {
                'Momentum': Momentum,
                'HighMoment': HighMoment,
                'Volatility': Volatility,
                'Liquidity': Liquidity,
                'Correlation': Correlation
            }
        self.analysis = analysis
        self.analysis_start_date = analysis_start_date
        self.num = num
        self.vecType = vecType
        self.beta_period = beta_period
        self.period_max = period_max
        return self

    def fillna_with_group(self, group):
        median = group.median()
        median.fillna(0, inplace=True)
        return group.fillna(median)

    def vec_data_process(self):
        data = self.source_data.copy()
        beta_p = self.beta_period
        max_p = self.period_max
        BTC = data['ret'].loc['BTCUSDT']
        BTC = pd.Series(BTC, name = 'BTC')
        data = data.join(BTC, on=data.index.get_level_values(1))
        data.drop('key_0', axis=1, inplace=True)
        calc_dataframe = data[['ret', 'BTC']]
        calc_tensor = np.zeros((data.shape[0], 2, beta_p))
        for i in range(beta_p):
            calc_tensor[:, :, i] = calc_dataframe.groupby(self.code_label).shift(i).values
        mean_tensor = np.mean(calc_tensor, axis=2)
        calc_tensor = calc_tensor-mean_tensor[:, :, np.newaxis]
        betas = (np.sum(calc_tensor[:,0,:]*calc_tensor[:,1,:], axis=1)/np.sum(calc_tensor[:,1,:]**2, axis=1))
        data['beta'] = betas
        data['Intercept'] = data['ret']-data['beta']*data['BTC']
        ew_mkt = data.groupby(self.time_label)['ret'].mean()
        ew_mkt = pd.Series(ew_mkt, name='ew_mkt')
        data = data.join(ew_mkt, on = data.index.get_level_values(1))
        data.drop('key_0', axis=1, inplace=True)
        data['total_q_vol'] = data.groupby(self.time_label)['quote_volume'].transform('sum')
        data['weight_return'] = data['ret'] * data['quote_volume'] / data['total_q_vol']
        qvol_w_mkt = data.groupby(self.time_label)['weight_return'].sum()
        qw_mkt = pd.Series(qvol_w_mkt, name='qw_mkt')
        data['total_close'] = data.groupby(self.time_label)['close'].transform('sum')
        data['close_w_return'] = data['ret'] * data['total_close'] / data['total_close']
        close_w_mkt = data.groupby(self.time_label)['close_w_return'].sum()
        cw_mkt = pd.Series(close_w_mkt, name='cw_mkt')
        data.drop(['total_q_vol', 'weight_return', 'total_close', 'close_w_return'], axis=1, inplace=True)
        data = data.join(qw_mkt, on = data.index.get_level_values(1))
        data.drop('key_0', axis=1, inplace=True)
        data = data.join(cw_mkt, on = data.index.get_level_values(1))
        data.drop('key_0', axis=1, inplace=True)
        col_list = data.columns.tolist()
        locations = {value: index for index, value in enumerate(col_list)}
        data_tensor = np.zeros((data.shape[0], data.shape[1], max_p))
        for i in range(max_p):
            data_tensor[:, :, i] = data.groupby(self.code_label).shift(i).values
        return data_tensor, locations


    def run(self, params):
        """将大类的key打在request params的head 用于识别对应大类里面去找func"""
        feature_queue = [[key] + value for key, value in params.items()]
        process = tqdm(feature_queue, desc='Feature Calculate Process')
        data = self.source_data.copy()
        if self.vecType == False:
            markets = self.markets_data.copy()
            for feature in process:
                func = self.feature_type_dict[feature[0]]()
                '''去掉class key变量'''
                cal_process = tqdm(feature[1:], desc = 'Feature Calculate Sub-Process')
                for rest in cal_process:
                    key, *args = rest
                    col_name = '_'.join(map(str, rest))
                    args = [markets[item] if item in markets else item for item in args]
                    fc = func.funcs[key]
                    data[col_name] = data.groupby(self.code_label).apply(lambda x: fc(x, *args)).reset_index(level=0, drop=True)
        elif self.vecType == True:
            data_tensor, locations = self.vec_data_process()
            for feature in process:
                func = self.feature_type_dict[feature[0]]()
                cal_process = tqdm(feature[1:], disable = True)
                for rest in cal_process:
                    key, *args = rest
                    col_name = '_'.join(map(str, rest))
                    args = [locations[item] if item in locations else item for item in args]
                    fc = func.funcs[key]
                    data[col_name] = fc(data_tensor, *args)

        if self.drop_raw:
            data.drop(columns=self.source_data.columns.tolist()[:-1], inplace=True)
        if self.fillna_media:
            data = data.groupby(self.time_label).apply(self.fillna_with_group).reset_index(level=0, drop=True)
        data['coin_old'] = data.groupby('jj_code').cumcount() + 1
        data = data[data['coin_old'] >= self.New_threshold]
        data.dropna(axis=0, how='any', inplace=True)
        factor_data = data[[col for col in data.columns if col != 'y'] + ['y']]
        factor_data.drop("BTCUSDT", inplace=True)
        if self.analysis:
            params = {
                "data": factor_data,
                "feature_names": factor_data.columns[:-1],
                "time_label": self.time_label,
                "code_label": self.code_label,
                "return_label": 'y',
                "frequency": self.frequency,
                "start_date": self.analysis_start_date
            }
            FA = FactorAnalysis().create(**params)
            FA.run(self.num)
        return factor_data
