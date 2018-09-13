import pandas as pd
import statsmodels.api as sm
from WindPy import *
import datetime
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import csv
from sklearn.ensemble import RandomForestRegressor

#第二步：预测收益或IC打分

#标准化去极值
def factor_clear_norm(data, cut=False, standard=True):
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


def debug_ind_code(code):
    if code!=code:
        return np.nan
    code = str(code)
    if code[:6]=='100000':
        dcode='b10h'
    elif code[:6]=='100002':
        dcode='b10t'
    else:
        dcode=code[:4]
    return dcode

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
    dummies = pd.get_dummies(df_factor['industry'])
    df_factor = pd.concat([df_factor, dummies], axis=1)
    Xlist = list(df_factor.columns)
    Xlist.remove('return')
    Xlist.remove('industry')

    df_factor=df_factor.dropna()

    #回归计算系数
    result = sm.OLS(df_factor['return'],df_factor[Xlist]).fit()
    params = result.params
    # print(result.rsquared)
    # P[date]=params
    # R[date]=[result.rsquared]
    return [params,result.rsquared]

#返回所选截面回归系数(下一期收益与当期因子值得回归系数)，return_period=20为累计20日收益率作为因变量
def linear_regression_WLS(factor_value,factor_name,return_value,date,return_period,industry):
    #读取当期的因子值和和收益率值
    params=[]
    market_size=pd.read_csv('mkt_cap_ard.csv',index_col=0)
    market_size=market_size.applymap(lambda x:math.sqrt(x))
    weight=market_size.ix[date,:]

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
    dummies = pd.get_dummies(df_factor['industry'])
    df_factor = pd.concat([df_factor, dummies], axis=1)
    Xlist = list(df_factor.columns)
    Xlist.remove('return')
    Xlist.remove('industry')

    df_factor['weight']=weight
    df_factor=df_factor.dropna()

    #回归计算系数
    result = sm.WLS(df_factor['return'],df_factor[Xlist],weights=df_factor['weight']).fit()
    params = result.params
    # print(result.rsquared)
    return [params,result.rsquared]

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
    print(datelist)
    return datelist


#通过线性回归模型，对截面预测下一期收益率（标准化后）
#H为半衰期，N为半衰期数，返回当期预测系数
def linear_params_avg(factor_value,factor_name,return_value,industry,date,H,N,return_period,method):

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
    Rsquared=pd.DataFrame(columns=['avg'])
    weight=[]

    # 获取每个截面的各因子系数
    for i in range(N):
        if method=='OLS':
            params=linear_regression(factor_value,factor_name,return_value,datelist[i],return_period,industry)[0]
            rsquared=linear_regression(factor_value,factor_name,return_value,datelist[i],return_period,industry)[1]
        elif method=='WLS':
            params = linear_regression_WLS(factor_value, factor_name, return_value, datelist[i], return_period, industry)[0]
            rsquared = linear_regression_WLS(factor_value, factor_name, return_value, datelist[i], return_period, industry)[1]
        Params[datelist[i]]=params
        Rsquared[datelist[i]]=[rsquared]
        if H==0:
            weight.append(1)
        else:
            weight.append((0.5 ** (1 / H)) ** i)
    #求系数的加权平均数
    Params['avg']=0
    Rsquared['avg']=0
    for i in range(N):
        Params['avg']+=weight[i]*Params[datelist[i]]
        Rsquared['avg']+=weight[i]*Rsquared[datelist[i]]
    Params['avg']=Params['avg']/sum(weight)
    Rsquared['avg']=Rsquared['avg']/sum(weight)
    print(date)
    # print(Params['avg'])
    # print(Rsquared['avg'])
    # P[date]=Params['avg']
    # R[date]=Rsquared['avg']
    return Params['avg']

def linear_predict_return(factor_value,factor_name,return_value,industry,date,H,N,return_period,regression_method):
    #计算当期预测系数
    params_avg=linear_params_avg(factor_value,factor_name,return_value,industry,date,H,N,return_period,regression_method)

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
    dummies = pd.get_dummies(df_factor['industry'])
    df_factor = pd.concat([df_factor, dummies], axis=1)
    df_factor['predict_return']=0
    for i in factor_name:#params_avg.index:
        df_factor['predict_return']+=df_factor[i]*params_avg[i]

    return df_factor['predict_return']

#返回当期截面因子的IC，return_period=20为收益率取累计20日收益率
def IC_score(factor_value,factor_name,return_value,date,return_period,industry):
    #读取当期的因子值和和收益率值
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
    dummies = pd.get_dummies(df_factor['industry'])
    df_factor = pd.concat([df_factor, dummies], axis=1)
    Xlist=list(df_factor.columns)
    Xlist.remove('industry')
    df_factor=df_factor.dropna()

    #计算IC值
    df_IC = df_factor[Xlist].corr()
    Xlist.remove('return')
    IC=df_IC.ix[Xlist,'return']
    return IC

#通过IC打分模型，对截面进行打分（标准化后）
#H为半衰系数，N为半衰期数，返回当期加权IC
def IC_score_avg(factor_value,factor_name,return_value,industry,date,H,N,return_period):

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

    Scores=pd.DataFrame(columns=['avg'])
    weight=[]

    # 获取每个截面的各因子IC
    for i in range(N):
        score=IC_score(factor_value,factor_name,return_value,datelist[i],return_period,industry)
        Scores[datelist[i]]=score
        if H==0:
            weight.append(1)
        else:
            weight.append((0.5 ** (1 / H)) ** i)

    #求IC的加权平均数
    Scores['avg']=0
    for i in range(N):
        Scores['avg']+=weight[i]*Scores[datelist[i]]
    Scores['avg']=Scores['avg']/sum(weight)
    # print(Scores['avg'])
    return Scores['avg']

def IC_predict_score(factor_value,factor_name,return_value,industry,date,H=2,N=2,return_period=20):
    #计算当期预测系数
    IC_avg=IC_score_avg(factor_value,factor_name,return_value,industry,date,H,N,return_period)

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
    dummies = pd.get_dummies(df_factor['industry'])
    df_factor = pd.concat([df_factor, dummies], axis=1)
    df_factor['predict_scores']=0
    for i in IC_avg.index:
        df_factor['predict_scores']+=df_factor[i]*IC_avg[i]

    return df_factor['predict_scores']


def randomtree_regression(factor_value, factor_name, return_value, industry, date, N, return_period):
    # 读取因子值和和收益率值
    # 因子个数
    tree_num = 50
    factor_num = len(factor_value)

    # 计算下一期收益
    future_ret = return_value.rolling(return_period).mean()
    future_ret = future_ret.shift(-return_period)

    trade_date = []
    for i in return_value.index:
        a = True
        for j in range(len(factor_value)):
            if i not in factor_value[j].index:
                a = False
        if (a):
            trade_date.append(i)
    date_list = get_date(date, return_period, N, trade_date)

    df_total = pd.DataFrame()
    for d in date_list:
        df_factor = pd.DataFrame()

        # 提取时间截面变量值
        df_factor['return'] = future_ret.ix[d, :]
        for i in range(factor_num):
            factor_date_value = factor_value[i].ix[d, :]
            df_factor[factor_name[i]] = factor_date_value
            # 标准去极值
            df_factor = factor_clear_norm(df_factor)

        # 增加行业因子
        df_factor['industry'] = industry
        dummies = pd.get_dummies(df_factor['industry'])
        df_factor = pd.concat([df_factor, dummies], axis=1)
        df_total = pd.concat([df_total, df_factor])
    df_total=df_total.dropna()
    Xlist =list(df_total.columns)
    Xlist.remove('return')
    Xlist.remove('industry')

    rfr = RandomForestRegressor(n_estimators=tree_num)
    rfr.fit(df_total[Xlist], df_total['return'])

    return rfr


def randomtree_predict(factor_value, factor_name, return_value, industry, date, N=2, return_period=20):
    # 训练模型
    rfr = randomtree_regression(factor_value, factor_name, return_value, industry, date, N, return_period)

    # 提取数据
    factor_num = len(factor_value)

    # 计算下一期收益
    future_ret = return_value.rolling(return_period).mean()
    future_ret = future_ret.shift(-return_period)

    df_factor = pd.DataFrame()

    # 提取时间截面变量值
    for i in range(factor_num):
        factor_date_value = factor_value[i].ix[date, :]
        df_factor[factor_name[i]] = factor_date_value
        # 标准去极值
        df_factor = factor_clear_norm(df_factor)

        # 增加行业因子
    df_factor['industry'] = industry
    dummies = pd.get_dummies(df_factor['industry'])
    df_factor = pd.concat([df_factor, dummies], axis=1)
    Xlist = list(df_factor.columns)
    df_factor=df_factor.dropna()
    Xlist.remove('industry')

    predict = pd.Series(rfr.predict(df_factor[Xlist]), index=df_factor.index)
    return predict

#第三步：回测

#判断交易当天是否停牌，输出截面与前一天未停牌股票list
def judge_suspended(date,susp_days):

    un_suspend=susp_days.ix[date,:][susp_days.ix[date,:]==0].index
    un_suspend_last=susp_days.shift(1).ix[date,:][susp_days.ix[date,:]==0].index
    return [un_suspend,un_suspend_last]

#判断前一交易日是否涨停与跌停,返回截面前一交易日未涨停股票和未跌停股票
def judge_limit(date,close_price):
    last_close=close_price.shift(1).ix[date,:]
    last_close_uplimit=round((close_price.shift(2).ix[date,:]*1.1),2)
    last_close_downlimit=round((close_price.shift(2).ix[date,:]*0.9),2)
    un_uplimit=last_close[last_close!=last_close_uplimit].index
    un_downlimit=last_close[last_close!=last_close_downlimit].index
    return [un_uplimit,un_downlimit]

#判断截面时是否已上市120日，返回上市超过120日的股票
def judge_new(date,ipo_date):
    deltadays=pd.Series(map(lambda x:(datetime.datetime.strptime(str(date),"%Y%m%d")-x).days,ipo_date),index=ipo_date.index)
    un_new=deltadays[deltadays>=120].index
    return un_new

#返回截面可卖股票和可买股票
def get_available_stock(date,susp_days,close_price,ipo_date):
    [un_suspend,un_suspend_last]=judge_suspended(date,susp_days)
    [un_uplimit, un_downlimit]=judge_limit(date,close_price)
    un_new=judge_new(date,ipo_date)
    buy_available=[]
    sell_available=[]
    for code in un_suspend:
        if code in un_new:
            if code in un_uplimit:
                buy_available.append(code)
    for code in un_suspend_last:
        if code in un_new:
            if code in un_downlimit:
                sell_available.append(code)

    return [buy_available,sell_available]


#通过收益率预测或者IC打分，结合可交易股票池,返回卖出交易信号
def sell_stock_select(sell_available,stock_account):
    sell_stock_list=stock_account
    sell_stock_list=sell_stock_list[sell_available]
    sell_stock_list[sell_stock_list==0]=np.nan
    sell_stock_list=sell_stock_list.dropna()
    return sell_stock_list

#通过收益率预测或者IC打分，结合可交易股票池,返回买入代码
def buy_stock_code(date,buy_available,factor_value,factor_name,return_value,industry,H,N,return_period,method,num,regression_method):
    if method=='linear':
        predict=linear_predict_return(factor_value,factor_name,return_value,industry,date,H,N,return_period,regression_method)
    elif method=='IC_score':
        predict=IC_predict_score(factor_value,factor_name,return_value,industry,date,H,N,return_period)
    elif method == 'Randomtree':
        predict = randomtree_predict(factor_value, factor_name, return_value, industry, date, N, return_period)
    buy_code=list(predict[buy_available].sort_values(ascending=False).index)[:num]
    return buy_code

def buy_stock_select(date,buy_available,cash_account,factor_value,factor_name,return_value,industry,H,N,return_period,method,num,open_price,regression_method):
    buy_code=buy_stock_code(date,buy_available,factor_value,factor_name,return_value,industry,H,N,return_period,method,num,regression_method)
    buy_stock_list=pd.Series(0,index=buy_code)
    for stock in buy_code:
        price=open_price.ix[date,stock]
        cash_available=(0.995*cash_account)/num
        buy_volume=int(cash_available/(100*price))*100
        buy_stock_list[stock]=buy_volume
    return buy_stock_list

#买入函数,都按限价买入计算
def order_buy(date, stock_code, stocknum, order_price, deal_table, slippery, cost,cash_account,stock_account,stock_adjust):

    #计算滑点、交易费用之后的成交价格,并入总的成交单表
    deal_cost=slippery*order_price*stocknum
    deal_table.append([date,stock_code,round(order_price,2),stocknum,'B'])

    #计算剩余现金
    cash_account=cash_account-order_price*stocknum-deal_cost

    #计算交易后的持仓
    stock_account.ix[stock_code,'volume']=stock_account.ix[stock_code,'volume']+stocknum
    stock_account.ix[stock_code,'buy_adjust']=stock_adjust.ix[date,stock_code]

    return [deal_table,cash_account,stock_account]


#卖出函数，都按限价卖出计算
def order_sell(date, stock_code, stocknum, order_price, deal_table, slippery,cost,cash_account,stock_account):

    #手续费计算
    order_price_2=order_price*stock_account.ix[stock_code,'now_adjust']/stock_account.ix[stock_code,'buy_adjust']
    deal_cost = slippery*order_price*stocknum
    deal_table.append([date,stock_code,round(order_price,2),stocknum,'S'])

    # 计算剩余现金
    cash_account=cash_account+order_price_2*stocknum-deal_cost

    # 计算交易后的持仓
    stock_account.ix[stock_code,'volume'] = stock_account.ix[stock_code,'volume'] - stocknum

    return [deal_table,cash_account,stock_account]

#计算截面账户净值
def my_account(date,cash_account,stock_account,close_price):
    market_value=0
    for stock in stock_account.index:
        if stock_account.ix[stock,'volume']==0:
            continue
        stock_close_price=close_price.ix[date,stock]
        market_value+=stock_close_price*stock_account.ix[stock,'volume']*stock_account.ix[stock,'now_adjust']/stock_account.ix[stock,'buy_adjust']
    account_value=market_value + cash_account
    date_account=pd.Series([cash_account,market_value,account_value],index=['cash_account','market_value','account_value'])
    return date_account

#返回调仓日期
def get_trade_date(start_date, end_date, return_period, tradeday_list):
    date_list=[]
    trade_day_list=[]
    for day in tradeday_list:
        if day>=start_date and day<=end_date:
            trade_day_list.append(day)
    for day in trade_day_list:
        num_j=trade_day_list.index(day)
        if(int((num_j)/return_period)==(num_j)/return_period):
            date_list.append(day)
    return [trade_day_list,date_list]

#在调仓截面进行交易
def back_test(initial_cash,start_date,end_date,return_period,factor_value,factor_name,return_value,H,N,num=20,method='linear',regression_method='OLS'):

    # 获取停牌、收盘价、开盘价、上市日期和行业数据
    susp_days = pd.read_csv('susp_days.csv', index_col=0)
    close_price = pd.read_csv('close.csv', index_col=0)
    close_price=close_price.fillna(method='pad')
    # print(close_price.ix[20170821,'155'])
    open_price=pd.read_csv('open.csv',index_col=0)
    tradeday_list=list(open_price.index)
    time_now = datetime.datetime.now()
    today = datetime.datetime.strftime(time_now, "%Y%m%d")
    code = w.wset("sectorconstituent", "date=" + today + ";sectorid=a001010100000000;field=wind_code").Data[0]
    ipo = w.wss(code, "ipo_date")
    code = map(lambda x: str(int(x[0:6])), code)
    code=list(code)
    ipo_date = pd.Series(ipo.Data[0], index=code, name=ipo.Fields[0])
    industry_code=pd.read_csv('industry_citiccode.csv',index_col=0)
    industry_code=industry_code.applymap(lambda x:debug_ind_code(x))
    stock_adjust=pd.read_csv('adjfactor.csv',index_col=0)
    stock_adjust=stock_adjust.fillna(method='pad')

    #初始化交易表，费率，现金账户和股票账户
    deal_table=[['date','code','price','volume','direction']]
    slippery=0.005
    cost=0
    cash_account=initial_cash
    total_account=pd.DataFrame(index=['cash_account','market_value','account_value'])
    stock_account = pd.DataFrame(index=code)
    stock_account['volume']=0
    stock_account['buy_adjust']=0
    stock_account['now_adjust'] = 0

    #获取调仓截面
    [trade_day_list,date_list]=get_trade_date(start_date,end_date,return_period,tradeday_list)
    # print(time.time() - t0)
    # trade_day_list=[20170821]
    for date in trade_day_list:
        stock_account['now_adjust']=stock_adjust.ix[date,:]
        if(date in date_list):

            # 计算可交易股票
            [buy_available, sell_available] = get_available_stock(date, susp_days, close_price, ipo_date)

            # 计算卖出信号
            sell_stock_list = sell_stock_select(sell_available, stock_account['volume'])

            #计算行业
            industry=industry_code.ix[date,:]

            # 先卖出股票
            for stock_code in sell_stock_list.index:
                sell_price = close_price.shift(1).ix[date, stock_code]
                [deal_table, cash_account, stock_account] = order_sell(date, stock_code, sell_stock_list[stock_code],
                                                                       sell_price, deal_table, slippery, cost,
                                                                       cash_account, stock_account)

            # 计算买入信号
            buy_stock_list = buy_stock_select(date, buy_available, cash_account, factor_value, factor_name,
                                              return_value, industry, H, N, return_period, method, num, open_price,regression_method)


            # 买入股票
            for stock_code in buy_stock_list.index:
                buy_price = open_price.ix[date, stock_code]
                [deal_table, cash_account, stock_account] = order_buy(date, stock_code, buy_stock_list[stock_code],
                                                                      buy_price, deal_table, slippery, cost,
                                                                      cash_account, stock_account,stock_adjust)
            # 所有交易完成后，计算账户净值


        date_account = my_account(date, cash_account, stock_account, close_price)
        total_account[date] = date_account
        print(time.time() - t0)
        print(date)
        # print(stock_account[stock_account != 0])
    total_account=total_account.T
    print(total_account)
    return [total_account,deal_table]

#第四步：计算并描述基金各指标

#绘制基金净值曲线
def get_plot(total_account):
    xs = [datetime.datetime.strptime(d, '%Y%m%d').date() for d in list(total_account.index.map(str))]
    plt.plot(xs, list(total_account['account_value']))
    plt.grid(True)
    plt.xlabel('Date')
    plt.ylabel('account_value')
    plt.show()

#返回夏普比率
def calculate_sharp_ratio(total_account,rate=0.04/252,date_num=252):
    # 计算日收益率(G3-G2)/G2
    total_account['return']=(total_account['account_value']-total_account['account_value'].shift(1))/total_account['account_value'].shift(1)

    # 计算超额回报率
    total_account['exReturn'] = total_account['return']-rate

    # 计算夏普比率
    sharperatio = math.sqrt(date_num) * total_account['exReturn'].mean() / total_account['exReturn'].std()
    return sharperatio

#返回最大回撤率和最大回撤区间
def max_drawdown(total_account):
    total_account['max']=np.maximum.accumulate(total_account['account_value'])
    total_account['max_drawdown']=(total_account['account_value']-total_account['max'])/total_account['max']

    # 最大回撤
    Max_drawdown=total_account['max_drawdown'].min()

    # 回撤结束时间点
    end_date=np.argmin(total_account['max_drawdown'])

    # 回撤开始的时间点
    start_date=np.argmax(total_account.ix[:end_date,'account_value'])
    return [Max_drawdown,start_date,end_date]

def summary(total_account):
    sharp_ratio=calculate_sharp_ratio(total_account=total_account)
    [Max_drawdown, start_date, end_date]=max_drawdown(total_account=total_account)
    print('夏普比率为： ',sharp_ratio)
    print('最大回撤率为： ',Max_drawdown)
    print('最大回撤发生时间为： ',start_date, end_date)
    get_plot(total_account=total_account)


#因子中性化
def netural(factor,mkt_netural=True):
    factor_netural = pd.DataFrame(index=factor.columns)
    industry_code = pd.read_csv('industry_citiccode.csv', index_col=0)
    industry_code = industry_code.applymap(lambda x: debug_ind_code(x))
    mkt = pd.read_csv('mkt_cap_ard.csv', index_col=0)
    mkt = mkt.applymap(lambda x: math.log(x + 1))
    for date in factor.index:
        if date < 20120409 or date > 20170931:
            continue
        df = pd.DataFrame()
        df['factor'] = factor.ix[date, :]
        if mkt_netural==True:
            df['mkt'] = mkt.ix[date, :]
        df = factor_clear_norm(df)
        df['industry'] = industry_code.ix[date, :]
        dummies = pd.get_dummies(df['industry'])
        df = pd.concat([df, dummies], axis=1)
        Xlist = list(df.columns)
        Xlist.remove('factor')
        Xlist.remove('industry')
        df = df.dropna()
        # 回归计算系数
        result = sm.WLS(df['factor'], df[Xlist]).fit()
        factor_netural[date] = result.resid
    factor_netural = factor_netural.T.fillna(method='pad')
    return factor_netural

w.start()
#读取因子和收益率
return_value=pd.read_csv('pct_chg.csv',index_col=0)
#第一步：数据清洗过程
industry_code=pd.read_csv('industry_citiccode.csv',index_col=0)
industry_code=industry_code.applymap(lambda x:debug_ind_code(x))
mkt=pd.read_csv('mkt_cap_ard.csv', index_col=0)
mkt=mkt.applymap(lambda x:math.log(x + 1))
turn=pd.read_csv('turn.csv', index_col=0)
turn_value=turn.rolling(20).sum()
turn_value_2=pd.read_csv('turn.csv',index_col=0)
turn_value_2[turn_value_2.ix[:,:]!=0]=1
turn_value_2=turn_value_2.rolling(20).sum()
turn_value_2[turn_value_2.ix[:,:]==0]=1
turn=turn_value/turn_value_2
momentum=pd.read_csv('pct_chg.csv', index_col=0)
momentum[momentum > 10]=10
momentum[momentum < -10]=-10
momentum=momentum.rolling(20).sum()
pb=pd.read_csv('pb_lf.csv',index_col=0)
pb[pb==0]=np.nan
pb=pb.fillna(method='pad')

#因子中性化
turn_netural=netural(turn)
momentum_netural=netural(turn)
pb_netural=netural(turn)
mkt_netural=netural(turn,False)
# print(turn_netural)
# print(pb_netural)
# print(mkt_netural)
# print(momentum_netural)

#回测
t0=time.time()
# P=pd.DataFrame()
# R=pd.DataFrame()
# trade_date_list=get_trade_date(20161001,20170931,20,mkt.index)[0]
# print(trade_date_list)
# IC_df=pd.DataFrame()
# for date in trade_date_list:
#     return_df=pd.DataFrame()
#     predict_return=linear_predict_return([mkt, pb, momentum, turn], ['mkt', 'pb', 'momentum', 'turn'],return_value,industry_code.ix[date,:],date,0,2,5,'OLS')
#     future_ret = return_value.rolling(5).mean()
#     future_ret = future_ret.shift(-5)
#     actual_return=future_ret.ix[date,:]
#     return_df['predict']=predict_return
#     return_df['actual']=actual_return
#     return_df['actual']=factor_clear_norm(return_df[['actual']])
#     IC=return_df.corr().ix['predict','actual']
#     IC_df[date]=[IC]
# IC_df.to_csv('IC_df2.csv')
# for day in mkt.index:
#     if day<20161001 or day>20170931:
#         continue
#     linear_params_avg([mkt, pb, momentum, turn], ['mkt', 'pb', 'momentum', 'turn'],
#                       return_value,industry_code.ix[day,:],day,0,2,20,'OLS')
[total_account1, deal_table1] = back_test(1000000, 20131001, 20170931, 5,[mkt_netural, pb_netural, momentum_netural, turn_netural], ['mkt', 'pb', 'momentum', 'turn'],return_value, 0, 5, 20, 'Randomtree', 'OLS')
summary(total_account1)
# linear_params_avg([mkt, pb, momentum, turn], ['mkt', 'pb', 'momentum', 'turn'],return_value,industry_code.ix[20171205,:],20171205,0,2,20,'linear')
# print(P)
# print(R)
# P.to_csv('P_2.csv')
# R.to_csv('R_2.csv')