# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 09:43:42 2021

@author: Richard Pan
"""
import time
import pymysql
from sqlalchemy import create_engine
import re
import pandas as pd
import numpy as np
import jieba
from nltk import FreqDist
from gensim import models,corpora 
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from wordcloud import WordCloud
from matplotlib import colors
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



#--------------------------------------导入数据----------------------------------#


def set_query(database,product):
    """
    用于创建SQL query语句
    
    Parameters
    ----------
    database : STRING
        数据库名称，自动带入.
    product : STRING
        产品名称，自动带入.

    Returns
    -------
    query : STRING
        MYSQL数据库查询语句.

    """
    if product == '全部':
        query = f'SELECT A.tid,A.created,B.seller_memo,A.title FROM {database}.t_order_refund A INNER JOIN {database}.t_order_base B ON A.tid=B.tid;'
    else:
        query = f'SELECT A.tid,A.created,B.seller_memo,A.title FROM {database}.t_order_refund A INNER JOIN {database}.t_order_base B ON A.tid=B.tid WHERE A.title="{product}";'
    return query



def get_message(txt):
    """
    用于提取message列中的有效文本

    Parameters
    ----------
    txt : STRIING
        来自DATAFRAME中message列的文本.

    Returns
    -------
    data : STRING
        提取到的有效文本，或可为空.

    """
    try:
        txt = re.findall('【.*】', txt)[0]
        txt = txt.replace('【','')
        txt = txt.replace('】','')
        data = re.findall('[\u4e00-\u9fa5]+',txt)
        data = ''.join(data)
        return data
    except:
        return ''



def getData(database,product,password,address,port,user):
    """
    【按钮】用于连接数据库并获取数据

    Parameters
    ----------
    database : STRING
        数据库名称
    product : STRING
        产品名称
    password : STRING
        数据库密码
    address : STRING
        数据库地址
    port : STRING
        数据库端口
    user : STRING
        数据库用户名

    Returns
    -------
    df_read : DATAFRAME
        返回读取到的数据，列为：ID,time,message,product.

    """
    engine = create_engine(f"mysql+pymysql://{database}:{password}@{address}:{port}/{user}")
    query = set_query(database, product)
    df_read = pd.read_sql_query(query, engine)
    df_read.columns = ['ID','time','message','product']
    df_read = df_read.set_index('ID')
    df_read['message'] = df_read['message'].apply(lambda x: get_message(x))
    return df_read

def get_data(bytes):
    """
    【按钮】读取上传的文件中的数据

    Parameters
    ----------
    file : link,STRING
        csv文件地址，如C://USER://file.csv.

    Returns
    -------
    df : DATAFRAME
        返回读取到的数据，列为：ID,time,message,product.

    """
    data = bytes.decode('utf-8').splitlines()
    df = pd.DataFrame(data)
    df.columns = ['ID','time','message','product']
    df = df.set_index('ID')
    df['message'] = df['message'].apply(lambda x: get_message(x))
    return df



#-------------------------------------添加词库-----------------------------------#


def add_words(ori_df,dic_df):
    """
    【勾选】将勾选的词库添加到默认词典中

    Parameters
    ----------
    ori_df : DATAFRAME
        原始词典数据.
    dic_df : DATAFRAME
        用于添加或删除的词库.

    Returns
    -------
    df : DATAFRAME
        原始词典数据 + or - 词库 后的词典数据.

    """
    df = pd.concat([ori_df,dic_df])
    return df


def drop_words(ori_df,dic_df):
    """
    【取消勾选】将取消勾选的词库从默认词典中删去

    Parameters
    ----------
    ori_df : DATAFRAME
        原始词典数据.
    dic_df : DATAFRAME
        用于添加或删除的词库.

    Returns
    -------
    df : DATAFRAME
        原始词典数据 + or - 词库 后的词典数据.

    """
    same = pd.merge(ori_df,dic_df, how='inner',on=['word'])
    df = pd.concat([ori_df,same]).drop_duplicates(keep=False)
    return df

def add_in(word,freq,tag):
    """

    Parameters
    ----------
    word : STRING
        来自词典word列的数据.
    freq : INTENGER
        来自词典freq列的数据.
    tag : STRING
        来自词典pos列的数据.

    Returns
    -------
    s : STRING
        用于执行添加词的语句.

    """
    s = f"jieba.add_word({word}, freq = {freq}, tag = {tag})"
    return s

def refresh_dic(df):
    """
    【按钮】修改默认词典

    Parameters
    ----------
    df : DATAFRAME
        最终要添加到默认词典中的词.

    Returns
    -------
    None.

    """
    for index, row in df.iterrows():
        exec(add_in(row['word'],row['freq'],row['pos']))



def split(txt,words_dic,stop_dic):
    """
    进行分词

    Parameters
    ----------
    txt : STRING
        来自数据message列的文本数据.
    words_dic: DATAFRAME
        添加的词典
    stop_dic : DATAFRAME
        停用词词库.

    Returns
    -------
    words : LIST
        分词结果列表.

    """
    refresh_dic(words_dic)
    stopwords_list = list(stop_dic['word'])
    sent = list(jieba.cut(txt,cut_all=False))
    words = []
    for word in sent:
        word = word.strip()
        if word not in stopwords_list:
            words.append(word)
    if len(words)>0:
        return words
    else:
        return np.nan



##需要添加同义词词库

def merge_word(df,same_dic):
    """
    合并同义词

    Parameters
    ----------
    df : DATAFRAME
        原始数据（以及完成分词）.
    same_dic : DICTIONARY
        替换同义词的字典.

    Returns
    -------
    df : DATAFRAME
        完成同义词替换的数据.

    """
    df['words'] = df['words'].apply(lambda x: [same_dic[i] if i in same_dic.keys() else i for i in x])
    return df


def split_words(df,words_dic,stop_dic,same_dic):
    """
    【按钮】执行分词

    Parameters
    ----------
    df : DATAFRAME
        等待分词的文本数据框.
    words_dic : DATAFRAME
        用于更新的词典.
    stop_dic : DATAFRAME
        停用词词典.
    same_dic : DICTIONARY
        同义词字典.

    Returns
    -------
    df : DATAFRAME
        完成分词的数据，新增列words.

    """
    df['words'] = df['message'].apply(lambda x: split(x,words_dic,stop_dic))
    df = merge_word(df,same_dic)
    return df



#----------------------------------建立词云-------------------------------------

def count_words(df,top):
    """
    词频计数

    Parameters
    ----------
    df : DATAFRAME
        分好词的数据，有words列.
    top : INT
        选取TOP X的词.

    Returns
    -------
    most_common_words : DATAFRAME
        DESCRIPTION.

    """
    data = df['words']
    all_words = []
    for i,v in data.items():
        all_words+=v
    fdisk = FreqDist(all_words)
    most_common_words = fdisk.most_common(top)
    return most_common_words


def draw_wc(df,top):
    """
    【按钮】生成词云，返回词频数据

    Parameters
    ----------
    df : DATAFRAME
        分好词的数据，有words列.
    top : INTENGER
        选取TOP X的词.

    Returns
    -------
    most_common_words : DATAFRAME
        TOP X的词以及它们的出现次数.

    """
    most_common_words = count_words(df,top)
    font_path = "C:/Windows/Fonts/" + "simsun.ttc"
    wc = WordCloud(
        font_path=font_path,
        background_color='white',
        contour_width=3,
        contour_color='steelblue',
        colormap='tab20',
        width=1000,
        height=1000
        ) 

    wc.generate_from_frequencies(dict(most_common_words))
    plt.figure(figsize=[50,50])
    plt.imshow(wc)
    plt.axis('off')
    plt.show()
    
    return most_common_words
    


#-----------------------------------主题模型-------------------------------------

def use_model(df,num_topics,num_words):
    """
    【按钮】生成主题模型

    Parameters
    ----------
    df : DATAFRAME
        用于生成模型的数据，已经完成分词，有words列.
    num_topics : INT
        主题的个数，即聚类的中心数.
    num_words : INT
        每个句子的主题数.

    Returns
    -------
    lsi : MODEL
        主题模型.
    topic : DICTIONARY
        主题字典.

    """
    dictionary = corpora.Dictionary(df['words'])
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in df['words']]
    tfidf = models.TfidfModel(doc_term_matrix)
    tf_matrix = tfidf[doc_term_matrix]
    lsi = models.LsiModel(tf_matrix, 
                          id2word=dictionary, 
                          num_topics=num_topics)
    topic = lsi.print_topics(num_words=num_words)
    return lsi,topic


def get_max(l):
    """
    判断主题，选取得分最高的结果

    Parameters
    ----------
    l : array
        使用模型，通过predict得到的结果.

    Returns
    -------
    result : STRING, key
        得分最高的主题.

    """
    d = dict(l)    
    result = max(d, key = lambda y: abs(float(d[y])))
    if abs(float(d[result])) >= 0.1:
        re = {}
        re['Key'] = result
        re['Value'] = d[result]
        return re

    else:
        result = {'Key':'NOT SURE','Value':0}
        return result


def test_model(test_data,model,topic,words_dic,stop_dic):
    """
    【按钮】获取主题聚类结果

    Parameters
    ----------
    test_data : DATAFRAME
        等待提取主题的数据.
    model : MODEL
        生成好的主题模型.
    topic : TUPLE LIST
        主题的元组列表.
    words_dic : DATAFRAME
        （之前已经确定）用于分词的新增词典.
    stop_dic : DATAFARME
        （之前已经确定）停用词词典.

    Returns
    -------
    test_data : DATAFRAME
        提取主题后的数据，新增'Theme_Code','Theme','Score'列，
        如果test_data是未经过分词的raw data,则额外再新增words列.

    """
    if 'words' not in test_data.columns:
        test_data['words'] = test_data['message'].apply(lambda x: split(x,words_dic,stop_dic))
        test_data = test_data[test_data['words'].apply(lambda x: False if x == [] else True)].reset_index(drop=True)
        test_data = merge_word(test_data['words'])
    dictionary = corpora.Dictionary(test_data['words'])
    test_data['matrix'] = test_data['words'].apply(lambda x:dictionary.doc2bow(x))
    test_data['predict'] = test_data['matrix'].apply(lambda x:model[x])
    test_data['Theme_Code'] = test_data['predict'].apply(lambda x: get_max(x)['Key'])
    dic = dict(topic)
    topics = {}
    for key,value in dic.items():
        topics[key] = value.split('"')[1].split('"')[0]
    test_data['Theme'] = test_data['Theme_code'].apply(lambda x: topics[x] if x!= 'NOT SURE' else 'NOT SURE')
    test_data['Score'] = test_data['predict'].apply(lambda x: abs(float(get_max(x)['Value'])))
    test_data = test_data[['ID','product','message','words','Theme_Code','Theme','Score','time']]
    return test_data
