# -*- coding: utf-8 -*-
"""
Created on Fri June 18 11:16:32 2021

@author: Richard Pan
"""

import pandas as pd
import numpy as np
import seaborn as sn
import datetime
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")


def getDateList(start_date, end_date):
    '''

    Parameters
    ----------
    start_date : STRING
        活动起始日期，如果有预热则为预热开始日期
    end_date : STRING
        活动结束日期

    Returns
    -------
    date_list : LIST
        返回介于start_date和end_date之间的所有日期列表

    '''
    date_list = []
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    date_list.append(start_date.strftime('%Y-%m-%d'))
    while start_date < end_date:
        start_date += datetime.timedelta(days=1)
        date_list.append(start_date.strftime('%Y-%m-%d'))
    
    return date_list

def get_special_day(special_day):
    '''

    Parameters
    ----------
    special_day : DATAFRAME
        记录了活动日期的数据集，包括活动名称、start time 和 end time

    Returns
    -------
    huodong_list : LIST
        返回所有Special Day的日期列表

    '''
    huodong_list = []
    for index,row in special_day.iterrows():
        date_list = getDateList(row['start time'],row['end time'])
        huodong_list+=date_list
    
    return huodong_list

def cal(data_grp,target):
    '''

    Parameters
    ----------
    data_grp : GROUPED DATAFRAME
        根据关键词groupby的dataframe
    target : STRING
        计算的目标，比如“点击率”、“展现量”

    Returns
    -------
    data : DATAFRAME
        返回完成计算的DATAFRAME

    '''
    data = pd.DataFrame()
    for name,group in data_grp:
        group = group.sort_values(by='日期',ascending=True)
        group[f'近3日{target}均值'] = group[target].rolling(window=3).mean().shift(1)
        group[f'近7日{target}均值'] = group[target].rolling(window=7).mean().shift(1)
        group[f'近15日{target}均值'] = group[target].rolling(window=15).mean().shift(1)
        group[f'近30日{target}均值'] = group[target].rolling(window=30).mean().shift(1)

        group[f'当日{target}对比3日均值'] = group[target]/group[f'近3日{target}均值']
        group[f'当日{target}对比7日均值'] = group[target]/group[f'近7日{target}均值']
        group[f'当日{target}对比15日均值'] = group[target]/group[f'近15日{target}均值']
        group[f'当日{target}对比30日均值'] = group[target]/group[f'近30日{target}均值']

        group = group.sort_values(by='日期',ascending=False)
        group[f'后30日{target}均值'] = group[target].rolling(window=30).mean().shift(1)
        group[f'后15日{target}均值'] = group[target].rolling(window=15).mean().shift(1)
        group[f'后7日{target}均值'] = group[target].rolling(window=7).mean().shift(1)
        group[f'后3日{target}均值'] = group[target].rolling(window=3).mean().shift(1)

        group[f'后3日{target}均值变动'] = group[f'后3日{target}均值']-group[target]
        group[f'后7日{target}均值变动'] = group[f'后7日{target}均值']-group[target]
        group[f'后15日{target}均值变动'] = group[f'后15日{target}均值']-group[target]
        group[f'后30日{target}均值变动'] = group[f'后30日{target}均值']-group[target]

        group[f'3日{target}均值单日变动比'] = group[[f'近3日{target}均值']].diff(-1,axis=0) / group[[f'近3日{target}均值']].shift(-1)
        group[f'7日{target}均值单日变动比'] = group[[f'近7日{target}均值']].diff(-1,axis=0) / group[[f'近7日{target}均值']].shift(-1)
        group[f'15日{target}均值单日变动比'] = group[[f'近15日{target}均值']].diff(-1,axis=0) / group[[f'近15日{target}均值']].shift(-1)
        group[f'30日{target}均值单日变动比'] = group[[f'近30日{target}均值']].diff(-1,axis=0) / group[[f'近30日{target}均值']].shift(-1)
        
        for column in list(group.columns[group.isnull().sum() > 0]):
            mean_val = group[column].mean()
            group[column].fillna(mean_val, inplace=True)
        for column in list(group.columns):
            max_val = group[column].max()
            group[column].replace(np.inf,max_val,inplace=True)
        
        data = data.append(group)
    
    data = data[['日期','SpecialDay','宝贝名称','关键词', '展现量', '点击量', '点击率', '投入产出比',
       f'近3日{target}均值', f'近7日{target}均值',f'近15日{target}均值', f'近30日{target}均值',
        f'当日{target}对比3日均值', f'当日{target}对比7日均值', f'当日{target}对比15日均值',f'当日{target}对比30日均值', 
       f'3日{target}均值单日变动比',f'7日{target}均值单日变动比', f'15日{target}均值单日变动比', f'30日{target}均值单日变动比',
        f'后30日{target}均值', f'后15日{target}均值', f'后7日{target}均值', f'后3日{target}均值',
       f'后3日{target}均值变动', f'后7日{target}均值变动', f'后15日{target}均值变动', f'后30日{target}均值变动',]]
    data.replace(-np.inf,0,inplace=True)
    data.replace(np.inf,0,inplace=True)
    data = data.dropna()
    return data

def test(data,word,target):
    d_test = data[data['关键词']==word]
    d_test_nonan = d_test.dropna()
    cor = d_test_nonan.corr()
    cor_3days = cor[f'后3日{target}均值']
    cor_7days = cor[f'后7日{target}均值']
    cor_30days = cor[f'后30日{target}均值']
    cor_15days = cor[f'后15日{target}均值']

    
    d_test_draw = d_test[['日期',target,f'近3日{target}均值',f'近7日{target}均值',f'近15日{target}均值', f'近30日{target}均值']]
    d_test_draw = d_test_draw.set_index('日期')
    myfont=FontProperties(fname=r'C:\Windows\Fonts\STSONG.TTF')
    sn.set(font=myfont.get_name())
    plt.figure(figsize=(30, 10))
    sn.lineplot(data = d_test_draw)
    plt.title(f'关键词：{word}')
    
    return cor_3days,cor_7days,cor_15days,cor_30days

def save_raw_data(data,file_name):
    data = data.replace(np.inf,np.nan)
    data.to_csv(f'{file_name}.csv')

def get_data(df,days,theme):
    '''

    Parameters
    ----------
    df : DATAFRAME
        原始数据，经过上面save_raw_data存储的已经计算、扩充好的数据集
    days : STRING
        字符串类型的数字，如"7"、"15"，决定预测未来days天的列名

    Returns
    -------
    df1 : DATAFRAME
        历史数据集
    df_predict : DATAFRAME
        用于预测的数据（每个词最新一天的表现，用来预测未来days天的表现）

    '''
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(by='日期',ascending=False)
    df1 = df[['日期','SpecialDay','宝贝名称','关键词','展现量', '点击量', '点击率', '投入产出比', f'近3日{theme}均值', f'近7日{theme}均值',
       f'近15日{theme}均值', f'近30日{theme}均值', f'当日{theme}对比3日均值', f'当日{theme}对比7日均值', f'当日{theme}对比15日均值',
       f'当日{theme}对比30日均值', f'3日{theme}均值单日变动比', f'7日{theme}均值单日变动比', f'15日{theme}均值单日变动比',
       f'30日{theme}均值单日变动比', f'后{days}日{theme}均值']]
    df1['月份'] = df['日期'].dt.month
    df1['几号'] = df['日期'].dt.day
    df1['周几'] = 1+df['日期'].dt.dayofweek
    df1 = df1[['宝贝名称','关键词','月份','几号','周几','SpecialDay','展现量', '点击量', '点击率', '投入产出比', f'近3日{theme}均值', f'近7日{theme}均值',f'近15日{theme}均值', f'近30日{theme}均值', f'当日{theme}对比3日均值', f'当日{theme}对比7日均值', f'当日{theme}对比15日均值',f'当日{theme}对比30日均值', f'3日{theme}均值单日变动比', f'7日{theme}均值单日变动比', f'15日{theme}均值单日变动比',f'30日{theme}均值单日变动比', f'后{days}日{theme}均值']]
    
    df_predict = df1.groupby(['宝贝名称','关键词']).head(1)
    df_predict = df_predict.fillna(0)
    df_predict[f'后{days}日{theme}均值'] = np.nan
    df1 = df1.dropna()
    df1 = df1.replace('inf',0)
    df1[['月份','几号','周几','SpecialDay','展现量', '点击量', '点击率', '投入产出比',f'近3日{theme}均值', f'近7日{theme}均值',
       f'近15日{theme}均值', f'近30日{theme}均值', f'当日{theme}对比3日均值', f'当日{theme}对比7日均值', f'当日{theme}对比15日均值',
       f'当日{theme}对比30日均值', f'3日{theme}均值单日变动比', f'7日{theme}均值单日变动比', f'15日{theme}均值单日变动比',
       f'30日{theme}均值单日变动比', f'后{days}日{theme}均值']] = df1[['月份','几号','周几','SpecialDay','展现量', '点击量', '点击率', '投入产出比', f'近3日{theme}均值', f'近7日{theme}均值',
       f'近15日{theme}均值', f'近30日{theme}均值', f'当日{theme}对比3日均值', f'当日{theme}对比7日均值', f'当日{theme}对比15日均值',
       f'当日{theme}对比30日均值', f'3日{theme}均值单日变动比', f'7日{theme}均值单日变动比', f'15日{theme}均值单日变动比',
       f'30日{theme}均值单日变动比', f'后{days}日{theme}均值']].astype('float64')
    df1 = df1.replace('inf',0)
    return df1,df_predict

def group_data(df):
    df = df.groupby(['宝贝名称','关键词'])
    return df

def LRM(name,x,y,x_pre):
    '''
    机器学习自动调参模型,下面的DTM、RFM、GBM、XGM、SVR同
    Parameters
    ----------
    name : STRING
        不需要指定，是自动带入的关键词
    x : DATAFRAME
        不需要指定，自动带入的关键词下的历史数据
    y : FLOAT
        不需要指定，自动带入的关键词后days日表现
    x_pre : TYPE
        不需要指定，自动带入的关键词最新日期的数据，用于预测

    Returns
    -------
    model : MODEL
        效果最佳的模型
    para : DICT
        最佳模型的参数
    R2 : FLOAT
        最佳模型的R2
    MSE : FLOAT
        最佳模型的MSE
    predict_target : ARRAY
        根据关键词最新一天的数据预测的未来表现

    '''
    m = Lasso()
    parameters = {'alpha': np.arange(0,1,0.1),
                  'fit_intercept':[True,False],
                  'max_iter':range(0,100,10),
                  'normalize':[True,False],
                  'precompute':[True,False],
                  'tol':[1e-3,1e-4,1e-5,1e-6],
                  'warm_start':[True,False],
                  'positive':[True,False],
                  'selection':["cyclic","random"]}
    clf = GridSearchCV(estimator=m,param_grid = parameters,n_jobs =-1,scoring='r2',cv=2)
    clf.fit(x,y)
    #print(pd.DataFrame(clf.cv_results_))
    model = clf.best_estimator_
    para = clf.best_params_
    R2 = clf.best_score_
    y_pred = model.predict(x)
    MSE = metrics.mean_squared_error(y, y_pred)
    predict_target = model.predict(x_pre)[0]
    return model,para,R2,MSE,predict_target

def DTM(name,x,y,x_pre):
    m = DecisionTreeRegressor()
    parameters = {'min_samples_split' : np.arange(0.11,1,0.1),
                  'min_samples_leaf' : np.arange(0,1,0.1),
                 'presort': [True,False]}
    clf = GridSearchCV(m,parameters,n_jobs =-1,scoring='r2',cv=2)
    clf.fit(x,y)
    model = clf.best_estimator_
    para = clf.best_params_
    R2 = clf.best_score_
    y_pred = model.predict(x)
    MSE = metrics.mean_squared_error(y, y_pred)
    predict_target = model.predict(x_pre)[0]
    return model,para,R2,MSE,predict_target

def RFM(name,x,y,x_pre):
    m = RandomForestRegressor()
    parameters = {'n_estimators': range(10,100,10),
                  'min_samples_split' : np.arange(0,1,0.1),
                  'min_samples_leaf' : np.arange(0,1,0.1),
                  'criterion':['mse','mae'],
                  'max_features':['auto','sqrt','log2'],
                    'bootstrap' : [False,True]}
    clf = GridSearchCV(m,parameters,n_jobs =-1,scoring='r2',cv=2)
    clf.fit(x,y)
    model = clf.best_estimator_
    para = clf.best_params_
    R2 = clf.best_score_
    y_pred = model.predict(x)
    MSE = metrics.mean_squared_error(y, y_pred)
    predict_target = model.predict(x_pre)[0]
    return model,para,R2,MSE,predict_target

def GBM(name,x,y,x_pre):
    m = GradientBoostingRegressor()
    parameters = {'n_estimators': range(10,100,10),
                  'min_samples_split' : np.arange(0,1,0.1),
                  'min_samples_leaf' : np.arange(0,1,0.1),
                  'criterion':['mse','mae'],
                 'learning_rate':np.arange(0,1,0.1),
                 'subsample':np.arange(0,1,0.1),
                 }
    clf = GridSearchCV(m,parameters,n_jobs =-1,scoring='r2',cv=2)
    clf.fit(x,y)
    model = clf.best_estimator_
    para = clf.best_params_
    R2 = clf.best_score_
    y_pred = model.predict(x)
    MSE = metrics.mean_squared_error(y, y_pred)
    predict_target = model.predict(x_pre)[0]
    return model,para,R2,MSE,predict_target

def XGM(name,x,y,x_pre):
    m = xgb.XGBRegressor()
    parameters = {'n_estimators':range(10,100,10),
                  'gamma': [0.1, 0.2, 0.3, 0.4, 0.5,],
                  'max_depth': [3, 4, 5], 
                  'min_child_weight': [1, 2, 3, 4, 5, 6],
                  'reg_alpha': [1, 2, 3, 4, 5], 
                  'reg_lambda': [1, 2, 3, 4, 5],
                  'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.07, 0.1, 0.2]
                 }
    clf = GridSearchCV(m,parameters,n_jobs =-1,scoring='r2',cv=2)
    clf.fit(x,y)
    model = clf.best_estimator_
    para = clf.best_params_
    R2 = clf.best_score_
    y_pred = model.predict(x)
    MSE = metrics.mean_squared_error(y, y_pred)
    predict_target = model.predict(x_pre)[0]
    return model,para,R2,MSE,predict_target

def SVRM(name,x,y,x_pre):
    m = SVR()
    parameters =  [{
        'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
        'gamma': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
        'kernel': ['rbf','sigmoid']},
        {
        'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
        'kernel': ['linear']
        },
        { 'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
          'gamma': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
         'degree':[2,3,4,5],
        'kernel': ['poily']}]
    clf = GridSearchCV(m,parameters,n_jobs =-1,scoring='r2',cv=2)
    clf.fit(x,y)
    model = clf.best_estimator_
    para = clf.best_params_
    R2 = clf.best_score_
    y_pred = model.predict(x)
    MSE = metrics.mean_squared_error(y, y_pred)
    predict_target = model.predict(x_pre)[0]
    return model,para,R2,MSE,predict_target

def write_result(name,model,para,R2,MSE,predict_column,predict_target):
    '''
    将所有类型的模型中的最佳模型写入汇总（之后再进行选择）
    Parameters
    ----------
    name : STRING
        关键词，无需指定，循环中自动带入
    model : MODEL
        模型，根据凋参得到的最佳模型
    para : DICT
        最佳模型的参数
    R2 : FLOAT
        最佳模型的R2
    MSE : FLOAT
        最佳模型的MSE
    predict_column : STRING
        预测数据的列名
    predict_target : ARRAY
        最佳模型计算的预测值

    Returns
    -------
    dic : DICT
        把模型、参数、R2、MSE、预测值打包成dict返回

    '''
    dic = dict()
    dic['宝贝名称'] = name[0]
    dic['关键词'] = name[1]
    dic['模型'] = model
    dic['参数'] = para
    dic['R2'] = R2
    dic['MSE'] = MSE
    dic[predict_column] = predict_target
    return dic

def choose_model(result):
    '''
    选择模型
    Parameters
    ----------
    result: DataFrame
        包含所有类型的模型的最佳模型

    Returns
    -------
    result : DataFrame
        返回该关键词最终选择的模型以及预测值

    '''
    result = result.sort_values(by=['MSE','R2'],ascending=(True,False))
    result = result.iloc[0,:]
    result = result.to_dict()
    return result

def multi_model(df,days,predict_column,x_pred,theme):
    '''

    Parameters
    ----------
    df : DATAFRAME
        原始的用于训练选模型的数据
    days : STRING
        预测未来days天的数据
    predict_column : STRING
        预测值存放的列的名称
    x_pred : DATAFRAME
        用于预测的X
    theme : STRING
        预测的主题，例如"曝光量"、"点击率"

    Returns
    -------
    model_df : DATAFRAME
        返回所有关键词的模型选型、参数、R2、MSE和预测出来的y

    '''
    y_target=f'后{days}日{theme}均值'
    model_df = []
    for name,group in df:
        result = []
        #print(group)
        group = group.drop(['宝贝名称','关键词'],axis=1)
        y = group[[y_target]]
        x = group.drop(y_target,axis=1)
        x_pre = x_pred[x_pred['宝贝名称'] == name[0]]
        x_pre = x_pre[x_pre['关键词']==name[1]]
        x_pre = x_pre.drop(['宝贝名称','关键词'],axis=1)
        x_pre = x_pre.drop(x_pre.columns[len(x_pre.columns)-1], axis=1)
        #--------------------------------------------------------------------------------------------#
        
        try:
            model,para,R2,MSE,predict_target = LRM(name,x,y,x_pre)
            dic = write_result(name,model,para,R2,MSE,predict_column,predict_target)
            result.append(dic)
            print('*-----------Lasso Finished-----------*')
            print(dic)
        except:
            print('LRM ERROR')
        
        try:
            model,para,R2,MSE,predict_target = DTM(name,x,y,x_pre)
            dic = write_result(name,model,para,R2,MSE,predict_column,predict_target)
            result.append(dic)
            print('*-----------Decision Tree Finished-----------*')
            print(dic)
        except:
            print('DTM ERROR')
        
        try:
            model,para,R2,MSE,predict_target = RFM(name,x,y,x_pre)
            dic = write_result(name,model,para,R2,MSE,predict_column,predict_target)
            result.append(dic)
            print('*-----------Random Forest Finished-----------*')
            print(dic)
        except:
            print('RFM ERROR')
        
        
        try:
            model,para,R2,MSE,predict_target = GBM(name,x,y,x_pre)
            dic = write_result(name,model,para,R2,MSE,predict_column,predict_target)
            result.append(dic)
            print('*-----------Gradient Boosting Finished-----------*')
            print(dic)
        except:
            print('GBM ERROR')
        
        '''
        try:
            model,para,R2,MSE,predict_target = XGM(name,x,y,x_pre)
            dic = write_result(name,model,para,R2,MSE,predict_column,predict_target)
            result.append(dic)
            print('*-----------XGBoosting Finished-----------*')
            print(dic)
        except:
            print('XGM ERROR')
       
        
        try:
            model,para,R2,MSE,predict_target = SVRM(name,x,y,x_pre)
            dic = write_result(name,model,para,R2,MSE,predict_column,predict_target)
            result.append(dic)
            print('*-----------SVR Finished-----------*')
            print(dic)
        except:
            print('SVRM ERROR')
       '''
        
        #--------------------------------------------------------------------------------------------#
        result = pd.DataFrame(result)
        result.to_csv('multi_model_test.csv')
        if result.empty:
            print(name)
            continue
        else:
            result = choose_model(result)
            print(result)
           
            #如果出现最好的模型R2小于0，则替换为近7日均值
            
            if float(result['R2'])<=0:
                result['参数'] = '模型无效，使用近30日均值'
                result['R2'] = 0
                result['MSE'] = np.nan
                result['模型'] = np.nan
                group = group.reset_index()
                result[predict_column] = group.loc[0,f'近30日{theme}均值']
            model_df.append(result)
            model_d = pd.DataFrame(model_df)
            model_d.to_csv('multi_model.csv')
        
    return model_df