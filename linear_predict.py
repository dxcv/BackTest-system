import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sqlalchemy import create_engine
import time
import pymysql

def factor_clear_norm(data, cut=False, standard=True):
    #标准化去极值
    midData=data
    for factor in midData.columns:
        m = midData[factor].median()
        mad = (midData[factor] - m).abs().median()
        dl = m - 3  * 1.483 * mad
        ul = m + 3  * 1.483 * mad

        #剪枝
        if cut == True:
            midData = midData[(midData[factor] >= dl) & (midData[factor] <= ul)]

        # 不剪枝
        else:
            midData.ix[midData[factor] <= dl, factor] = dl
            midData.ix[midData[factor] >= ul, factor] = ul

        #标准化
        if standard == True:
            midData[factor] = (midData[factor] - midData[factor].mean()) / midData[factor].std()
    return midData

def connect_jh_db(API='sqlalchemy', echo=False):
    User = 'guoyupeng'
    DBName = 'quant_db'
    Password = 'Abcd1234'
    IP = '192.168.1.11'
    if API == 'sqlalchemy':
        engine = create_engine("mysql+pymysql://%s:%s@%s:3306/%s?charset=utf8" % (User, Password, IP, DBName),
                               echo=echo, encoding="utf-8")
        return engine
    elif API == 'pymysql':
        conn = pymysql.connect(host=IP, port=3306, user=User, passwd=Password, db=DBName, charset='utf8')
        return conn

#返回行业变量
def get_industry():
    engine = connect_jh_db(API='pymysql')
    industry_factor = pd.read_sql_query('select wind_code,ind_code from industry', engine).dropna()
    industry_factor = industry_factor.drop_duplicates('wind_code')
    industry_factor = industry_factor.set_index('wind_code')
    industry_factor.index = map(str, industry_factor.index)
    industry_factor = industry_factor.applymap(lambda x: x[0:4])
    return industry_factor

#返回各因子值和日收益率值
def get_data():
    #读取因子数据和收益率数据
    factor1=pd.DataFrame()
    factor2=pd.DataFrame()
    factor_value=[factor1,factor2]
    return_value=pd.DataFrame()
    #数据清洗步骤

    return [factor_value,return_value]

#返回所选截面回归系数(下一期收益与当期因子值得回归系数)，return_period=20为累计20日收益率作为因变量
def linear_regression(factor_value,factor_name,return_value,date,return_period,industry):
    #读取当期的因子值和和收益率值
    params=[]

    #因子个数
    factor_num=len(factor_value)

    #计算下一期收益
    future_ret = return_value.rolling(return_period).mean()
    future_ret = future_ret.shift(-return_period)
    df_factor=pd.DataFrame()

    #提取时间截面变量值
    df_factor['return']=future_ret.ix[date,:]
    for i in range(factor_num):
        factor_date_value=factor_value[i].ix[date,:]
        df_factor[factor_name[i]]=factor_date_value

    #标准去极值
    df_factor=factor_clear_norm(df_factor)

    #增加行业因子
    df_factor['industry']=industry
    dummies = pd.get_dummies(df_factor['industry'], drop_first=True)
    df_factor = pd.concat([df_factor, dummies], axis=1)
    Xlist = list(df_factor.columns)
    Xlist.remove('return')
    Xlist.remove('industry')

    df_factor=df_factor.dropna()

    #回归计算系数
    result = sm.OLS(df_factor['return'],df_factor[Xlist]).fit()
    params = result.params
    return params

#计算会用到的前日期截面
def get_date(date,return_period,N,trade_date):
    datelist=[]
    num_i=trade_date.index(date)
    for i in range(N):
        if num_i-(N-i)*return_period>=0:
            datelist.append(trade_date[num_i-(N-i)*return_period])
        else:
            print("输入日期区间与半衰期数超出限制")
            assert num_i-(N-i)*return_period>=0
    return datelist


#通过线性回归模型，对截面预测下一期收益率（标准化后）
#H为半衰期，N为半衰期数，返回当期预测系数
def linear_params_avg(factor_value,factor_name,return_value,industry,date,H,N,return_period):

    #datelist为前几期截面，长度为N,排序为顺序
    trade_date=[]
    for i in return_value.index:
        a=True
        for j in range(len(factor_value)):
            if i not in factor_value[j].index:
                a=False
        if(a):
            trade_date.append(i)
    datelist=get_date(date,return_period,N,list(trade_date))

    Params=pd.DataFrame(columns=['avg'])
    weight=[]

    # 获取每个截面的各因子系数
    for i in range(N):
        params=linear_regression(factor_value,factor_name,return_value,datelist[i],return_period,industry)
        Params[datelist[i]]=params
        weight.append(H**i)

    #求系数的加权平均数
    Params['avg']=0
    for i in range(N):
        Params['avg']+=weight[i]*Params[datelist[i]]
    Params['avg']=Params['avg']/sum(weight)

    return Params['avg']

def linear_predict_return(factor_value,factor_name,return_value,industry,date,H=2,N=2,return_period=20):
    #计算当期预测系数
    params_avg=linear_params_avg(factor_value,factor_name,return_value,industry,date,H,N,return_period)

    #提取当日因子值
    factor_num=len(factor_value)
    df_factor=pd.DataFrame()
    for i in range(factor_num):
        factor_date_value=factor_value[i].ix[date,:]
        df_factor[factor_name[i]]=factor_date_value

    #标准去极值
    df_factor=factor_clear_norm(df_factor)

    #增加行业因子
    df_factor['industry']=industry
    dummies = pd.get_dummies(df_factor['industry'], drop_first=True)
    df_factor = pd.concat([df_factor, dummies], axis=1)
    df_factor['predict_return']=0
    for i in params_avg.index:
        df_factor['predict_return']+=df_factor[i]*params_avg[i]

    return df_factor['predict_return']


# t0=time.time()
# df1=pd.read_csv('Amkt_linear2.csv',index_col=0)
# df3=pd.read_csv('ATOT_WRATING.csv',index_col=0)
# df2=pd.read_csv('pct_chg.csv',index_col=0)
# m=get_industry()
# print(time.time()-t0)
# aa=linear_predict_return([df1,df3],['Amkt','ATOT'],df2,m,20150413,2,3,20)
# print(time.time()-t0)
# b=get_date(20120409,20,3,list(df1.index))
# print(aa)