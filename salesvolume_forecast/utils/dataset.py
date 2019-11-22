# -*- coding:utf8 -*-
import sys
import numpy as np
import pandas as pd
import os 
import gc
import math
from tqdm import tqdm, tqdm_notebook
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import datetime
import time
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import BaggingRegressor
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

def read_data(path):
    # path  = 'E:/CarSalesPrediction/'
    train_sales  = pd.read_csv(path+'train2_dataset/train_sales_data.csv', 
        converters={'salesVolume': lambda u: np.log1p(float(u)) if float(u) > 0 else 0})
    train_search = pd.read_csv(path+'train2_dataset/train_search_data.csv',
        converters={'popularity': lambda u: np.log1p(float(u)) if float(u) > 0 else 0})
    train_user_reply = pd.read_csv(path+'train2_dataset/train_user_reply_data.csv',
        converters={'carCommentVolum': lambda u: np.log1p(float(u)) if float(u) > 0 else 0,
        'newsReplyVolum': lambda u: np.log1p(float(u)) if float(u) > 0 else 0})
    evaluation_public = pd.read_csv(path+'test2_dataset/evaluation_public.csv')
    # submit_example    = pd.read_csv(path+'submit_example.csv')
    data = pd.concat([train_sales, evaluation_public], ignore_index=True)
    data = data.merge(train_search, 'left', on=['province', 'adcode', 'model', 'regYear', 'regMonth'])
    data = data.merge(train_user_reply, 'left', on=['model', 'regYear', 'regMonth'])
    # data['label'] = data['salesVolume']
    data['id'] = data['id'].fillna(0).astype(int)
    data['bodyType'] = data['model'].map(train_sales.drop_duplicates('model').set_index('model')['bodyType'])

    for i in ['bodyType', 'model']:
        data[i] = data[i].map(dict(zip(data[i].unique(), range(data[i].nunique()))))
    data['delta_month'] = (data['regYear']-2016)*12+data['regMonth']

    tmp = data['salesVolume'].groupby([data['adcode'], data['regYear'], data['regMonth']]).sum().reset_index()
    tmp.rename(columns={'salesVolume':'adcode_year_month_sales_all'}, inplace=True)
    data = data.merge(tmp, 'left', on=['adcode', 'regYear', 'regMonth'])
    data['adcode_year_month_sales_all'] = data['adcode_year_month_sales_all']-data['salesVolume']

    tmp = data['salesVolume'].groupby([data['adcode'], data['regYear'], data['regMonth'], data['bodyType']]).sum().reset_index()
    tmp.rename(columns={'salesVolume':'adcode_year_month_bodyType_sales_all'}, inplace=True)
    data = data.merge(tmp, 'left', on=['adcode', 'regYear', 'regMonth', 'bodyType'])
    data['adcode_year_month_bodyType_sales_all'] = data['adcode_year_month_bodyType_sales_all']-data['salesVolume']

    tmp = data['salesVolume'].groupby([data['model'],data['regYear'], data['regMonth']]).sum().reset_index()
    tmp.rename(columns={'salesVolume':'model_year_month_sales_all'}, inplace=True)
    data = data.merge(tmp, 'left', on=['model', 'regYear', 'regMonth'])
    data['model_year_month_sales_all'] = data['model_year_month_sales_all']-data['salesVolume']
    
    day_map = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
    data['dayCount'] = data['regMonth'].map(day_map)
    data.loc[(data.regMonth==2)&(data.regYear==2016),'dayCount']=29
    data['salesVolumePerday'] = data['salesVolume']/data['dayCount']

    return data

def get_feature(data, T):
    df = data.copy()
    # scaler = StandardScaler()
    # scaler.fit(pd.concat([X_train]))
    # X_train[:] = scaler.transform(X_train)
    # feature = ['salesVolume','popularity','carCommentVolum','newsReplyVolum',\
               # 'adcode_year_month_sales_all','adcode_year_month_bodyType_sales_all','model_year_month_sales_all']
    feature = []
    y = []
    df['model_adcode'] = df['adcode'] + df['model']
    df['model_adcode_mt'] = df['model_adcode']*100 + df['delta_month']
    for col in tqdm(['salesVolumePerday', 'salesVolume','popularity','carCommentVolum','newsReplyVolum',\
               'adcode_year_month_sales_all','adcode_year_month_bodyType_sales_all','model_year_month_sales_all']):
        # history feature
        for i in range(1,T+1):
            feature.append('shift_model_adcode_mt_{}_{}'.format(col,i))
            df['model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'] + i
            df_last = df[~df[col].isnull()].set_index('model_adcode_mt_{}_{}'.format(col,i))
            df['shift_model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'].map(df_last[col])
    
    for col in tqdm(['salesVolumePerday', 'salesVolume','popularity','carCommentVolum','newsReplyVolum',\
               'adcode_year_month_sales_all','adcode_year_month_bodyType_sales_all','model_year_month_sales_all']):
        # history feature delta
        for pair in [(1,2), (1,3), (2,3), (2,4), (3,4), (3,5), (4,5), (4,6), (1,6), (1,T)]:
            feature.append('rise_model_adcode_mt_{}_{}_{}'.format(col,pair[0],pair[1]))
            df['rise_model_adcode_mt_{}_{}_{}'.format(col,pair[0],pair[1])] = df['shift_model_adcode_mt_{}_{}'.format(col,pair[0])] \
                                               - df['shift_model_adcode_mt_{}_{}'.format(col,pair[1])]
            # print(df['rise_model_adcode_mt_{}_{}_{}'.format(col,pair[0],pair[1])])
            # exit()
            # df['ratio_model_adcode_mt_{}_{}_{}'.format(col,pair[0],pair[1])].round(5).to_csv('test_{}_{}_{}.csv'.format(col,pair[0],pair[1]), index=False)
    
    for col in tqdm(['salesVolumePerday', 'salesVolume', 'adcode_year_month_sales_all','adcode_year_month_bodyType_sales_all','model_year_month_sales_all']):
        # history feature delta
        for pair in [(1,2), (1,3), (2,3), (2,4), (3,4), (3,5), (2,6), (1,6), (1,T)]:
            feature.append('ratio_model_adcode_mt_{}_{}_{}'.format(col,pair[0],pair[1]))
            df['ratio_model_adcode_mt_{}_{}_{}'.format(col,pair[0],pair[1])] = df['shift_model_adcode_mt_{}_{}'.format(col,pair[0])] \
                                               / (df['shift_model_adcode_mt_{}_{}'.format(col,pair[1])])

    for col in tqdm(['salesVolume']):
        for i in [0,1,2,3]:
            y.append('after_model_adcode_mt_{}_{}'.format(col,i))
            df['model_adcode_mt_{}_{}'.format(col,-i)] = df['model_adcode_mt'] - i
            df_last = df[~df[col].isnull()].set_index('model_adcode_mt_{}_{}'.format(col,-i))
            df['after_model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'].map(df_last[col])

    feature.extend(['bodyType', 'model', 'adcode'])
    
    return df, feature, y

def get_Xy(df, T, features, y):
    train_idx = (df['delta_month'].between(T+1, 20))
    valid_idx = (df['delta_month'].between(21, 21))
    test_idx  = (df['delta_month'].between(25, 25))

    train_x = df[train_idx][features]
    train_y = df[train_idx][y]
    valid_x = df[valid_idx][features]
    valid_y = df[valid_idx][y]
    valid_model = df[valid_idx]['model'].values
    length_valid = len(valid_model)
    model2index = {}
    for i in range(length_valid):
        if valid_model[i] not in model2index:
            model2index[valid_model[i]] = []
        model2index[valid_model[i]].append(i)

    test_x = df[test_idx][features]

    return train_x, train_y.values, valid_x, valid_y.values, test_x, model2index

def normalization(X_train, X_val, X_test):
    scaler = StandardScaler()
    scaler.fit(pd.concat([X_train, X_val, X_test]))
    X_train[:] = scaler.transform(X_train)
    X_val[:] = scaler.transform(X_val)
    X_test[:] = scaler.transform(X_test)

    X_train = X_train.as_matrix()
    X_val = X_val.as_matrix()
    X_test = X_test.as_matrix()

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    return X_train, X_val, X_test

