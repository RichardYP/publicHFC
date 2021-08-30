# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 10:22:26 2021

@author: Y.P
"""

import os
import pandas as pd

path = os.getcwd()
path_ic = path + '\\' +  'IC_daily'
path_if = path + '\\' +  'IF_daily'
path_id = path + '\\' +  'ID_daily'
path_bm = path + '\\' +  'BenchMarking'
path_ff = path + '\\' +  'FundFocus'
path_alpha = path + '\\' +  'Alpha\\'

def trans_to_df(file):
    """
    Transfer csv to dataframe,to get target funds' stocks and their percent
    Parameters
    ----------
    file : STRING
        file path.

    Returns
    -------
    output : DataFrame
        Transfered Dataframe.

    """
    with open(file) as f:
        data = f.read()
    
    data = data.split('\n')
    data = data[5:]
    output = [['SecuCode','Type','Share%InIndex']]
    for d in data:
        if d != '':
            d = d.split(' ')
            #print(d)
            output.append(d)
    output = pd.DataFrame(output[1:],columns=output[0])
    output['Share%InIndex'] = output['Share%InIndex'].apply(lambda x: float(x)*100)
    output.set_index('SecuCode',inplace=True)
    #print(output)
    return output

'''
Example: 
file = path_ic + '\\IC_daily.20191231'
trans_to_df(file)
'''

def get_refer(file):
    """
    Read and transfer refer file, get CSI 300 and CSI 500 stocks and their percent
    Parameters
    ----------
    file : STRING
        File Path.

    Returns
    -------
    data : DataFrame
        Transfered DataFrame.

    """
    data = pd.read_csv(file,encoding='gbk')
    data = data.loc[(data['FundRefer']=='中证500指数') | (data['FundRefer']=='沪深300指数') | (data['FundRefer']=='中证800指数')]
    data = data.loc[(data['FundType']=='混合型') | (data['FundType']=='股票型')]
    data = data[data['FundName'].str.contains('量化')==False]
    data['FundCode'] = data['FundCode'].apply(lambda x: str(x).zfill(6))
    data.set_index('FundCode',inplace=True)
    print(data)
    return data
    
'''
Example:
file = 'Fund_List_Refer20210109.csv'
get_refer(file)
'''

def match(file,refer):
    """
    Main function, match and merge two files
    Parameters
    ----------
    file : STRING
        file path.
    refer : STRING
        refer file path.

    Returns
    -------
    Output : DataFrame
        About all stocks both in CSI300+CSI500 and target funds, including their percent and Size.

    """
    FundCode = file.split('_')[0]
    TradingDay = file.split('_')[1]
    TD = TradingDay.replace('-','')
    file = path_ff + '\\' + file
    Input = pd.read_csv(file,encoding='utf-8-sig')
    Input['SecuCode'] = Input['SecuCode'].apply(lambda x: str(x).zfill(6))
    Input.set_index('SecuCode',inplace=True)
    #print(Input)
    try:
        idx = refer.loc[FundCode,'FundRefer']
    except:
        Output = pd.DataFrame()
        return(Output)
        pass
    #print(idx)
    if idx=='中证500指数':
        fl = path_ic + '\\' + f"IC_daily.{TD}"
        Index = trans_to_df(fl)
    elif idx=='沪深300指数':
        fl = path_if + '\\' + f"IF_daily.{TD}"
        Index = trans_to_df(fl)
    elif idx=='中证800指数':
        fl = path_id + '\\' + f"CSI800_daily.{TD}"
        Index = trans_to_df(fl)
    else:
        Output = pd.DataFrame()
        return(Output)
        pass
    
    Output = pd.DataFrame()
    for index,row in Input.iterrows():
        try:
            row['Share%InIndex'] = Index.loc[index,'Share%InIndex']
        except:
            row['Share%InIndex'] = 0.00
        row['Share_Surplus'] = float(row['Share%InStock']) - float(row['Share%InIndex'])
        Output = Output.append(row)
    
    Output = Output.reset_index()
    try:
        Output.columns = ['SecuCode', 'Amount', 'FundCode', 'FundName', 'SecuName', 'Share%','Share%InIndex', 'Share%InStock', 'Share_Surplus', 'Size','UpdateTime']
        return Output
    except:
        print(file)

def save_file(file,output):
    """
    Save files
    Parameters
    ----------
    file : String
        File name.
    output : DataFrame
        result of match.

    Returns
    -------
    Alpha : DataFrame
        Some columns will be used for calculate later.

    """
    try:
        BenchMarking = output[['FundCode', 'FundName','UpdateTime','SecuCode', 'SecuName','Amount','Size','Share%', 'Share%InStock', 'Share%InIndex','Share_Surplus']]    
        path = path_bm + '\\' + file
        BenchMarking.to_csv(path,encoding='utf-8-sig',index=False)
        
        Alpha = output[['UpdateTime','FundCode','SecuCode','Size','Share_Surplus']]
        Alpha.columns =['TradingDay','FundCode','SecuCode','Size','Share_Surplus']
        #print(Alpha)
        return Alpha
    except:
        print(file)



def Alpha_size(Alpha):
    """
    Size factor
    Parameters
    ----------
    Alpha : DataFrame
        Result of save_file.

    Returns
    -------
    output : DataFrame
        time, stockcode and alpha value.

    """
    output = pd.DataFrame()
    Alpha = Alpha.groupby(['TradingDay','SecuCode'])
    for name,group in Alpha:
        total = group['Size'].sum()
        #print(total)
        group['Weighted_Surplus'] = group[['Share_Surplus','Size']].apply(lambda x: (x['Size']/total)*x['Share_Surplus'],axis=1)
        #print(group)
        output = output.append(group)
    output = output[['TradingDay','SecuCode','Weighted_Surplus']]
    output = output.groupby(['TradingDay','SecuCode']).sum()
    #output.to_csv('test.csv',encoding='utf-8-sig')
    output = output.reset_index()
    output.columns = ['TradingDay','SecuCode','Weighted_Surplus']
    #print(output)
    return output
    

def Alpha_med(Alpha):
    """
    Median factor
    Parameters
    ----------
    Alpha : DataFrame
        Result of save_file.

    Returns
    -------
    output : DataFrame
        time, stockcode and alpha value.

    """
    Alpha = Alpha[['TradingDay','SecuCode','Share_Surplus']]
    Alpha = Alpha.groupby(['TradingDay','SecuCode']).median()
    Alpha = Alpha.reset_index()
    Alpha.columns = ['TradingDay','SecuCode','Median_Surplus']
    #print(Alpha)
    return Alpha
    

def Alpha_rank(Alpha):
    """
    Rank factor
    Parameters
    ----------
    Alpha : DataFrame
        Result of save_file.

    Returns
    -------
    output : DataFrame
        time, stockcode and alpha value.

    """
    rank_weight = pd.DataFrame()
    rank_weight['Top%'] = [10,20,30,40,50,60,70,80,90,100]
    rank_weight['10/top%'] = rank_weight['Top%'].apply(lambda x: 10/x)
    all_w = rank_weight['10/top%'].sum()
    rank_weight['Weight'] = rank_weight['10/top%'].apply(lambda x: x/all_w)
    print(rank_weight)

def Alpha_whole_size(Alpha):
    """
    Whole size factor
    Parameters
    ----------
    Alpha : DataFrame
        Result of save_file.

    Returns
    -------
    output : DataFrame
        time, stockcode and alpha value.

    """
    output = pd.DataFrame()
    Alpha = Alpha.groupby('TradingDay')
    for name,group in Alpha:
        total = group['Size'].sum()
        #print(total)
        group['Whole_Weighted_Surplus'] = group[['Share_Surplus','Size']].apply(lambda x: (x['Size']/total)*x['Share_Surplus'],axis=1)
        #print(group)
        output = output.append(group)
    output = output[['TradingDay','SecuCode','Whole_Weighted_Surplus']]
    output = output.groupby(['TradingDay','SecuCode']).sum()
    #output.to_csv('test.csv',encoding='utf-8-sig')
    output = output.reset_index()
    output.columns = ['TradingDay','SecuCode','Whole_Weighted_Surplus']
    #print(output)
    return output

def filter_top(file_list,p):
    """
    Get TOP P stocks
    Parameters
    ----------
    file_list : STRING
        File path, include all history matched data.
    p : INT
        Top p.

    Returns
    -------
    output : List
        Top P stocks' code list.

    """
    new_list = []
    for file in file_list:
        file = file.split('_')
        new_list.append(file)
    
    new_list = pd.DataFrame(new_list,columns=['FundCode','ReportDate','Growth%'])
    total = new_list.shape(1)
    target = int(total*p)
    new_list.sort_values('Growth%')
    output = new_list.head(target)
    return output



"""
Example:

f = 'Fund_List_Refer20210109.csv'
refer = get_refer(f)

Alpha = pd.DataFrame()
file_list = os.listdir(path_ff)
for file in file_list:
    #file = '000006_2019-12-31_0.21199999999999997.csv'
    output = match(file, refer)
    if output.empty:
        continue
    else:
        Ap = save_file(file,output)
        Alpha = Alpha.append(Ap)

#print(Alpha)


#file_list = os.listdir(path_bm)
#Alpha = pd.DataFrame()
#for file in file_list:
#    file = path_bm + '\\' + file
#    output = pd.read_csv(file)
#    Ap = output[['UpdateTime','FundCode','SecuCode','Size','Share_Surplus']]
#    Ap.columns =['TradingDay','FundCode','SecuCode','Size','Share_Surplus']
#    Alpha = Alpha.append(Ap)



#AlphaSize = Alpha_size(Alpha)
#print(AlphaSize)
#AlphaSize.to_csv(path_alpha+'Alpha_Size.csv',encoding='utf-8-sig',index=False)

#AlphaMed = Alpha_med(Alpha)
#print(AlphaMed)
#AlphaMed.to_csv(path_alpha+'Alpha_Median.csv',encoding='utf-8-sig',index=False)

#AlphaSize = Alpha_whole_size(Alpha)
#print(AlphaSize)
#AlphaSize.to_csv(path_alpha+'Alpha_Size.csv',encoding='utf-8-sig',index=False)

#Alpha_rank(Alpha)

"""
