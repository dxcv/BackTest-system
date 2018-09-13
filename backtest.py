import pandas as pd
from WindPy import *
# import time

w.start()

#判断交易当天是否停牌，输出截面未停牌股票list
def judge_suspended(date,susp_days):

    un_suspend=susp_days.ix[date,:][susp_days.ix[date,:]==0].index
    return un_suspend

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
    deltadays=pd.Series(map(lambda x:(datetime.strptime(str(date),"%Y%m%d")-x).days,ipo_date),index=ipo_date.index)
    un_new=deltadays[deltadays>=120].index
    return un_new

#返回截面可卖股票和可买股票
def get_available_stock(date,susp_days,close_price,ipo_date):
    un_suspend=judge_suspended(date,susp_days)
    [un_uplimit, un_downlimit]=judge_limit(date,close_price)
    un_new=judge_new(date,ipo_date)
    buy_available=[]
    sell_available=[]
    for code in un_suspend:
        if code in un_new:
            if code in un_uplimit:
                buy_available.append(code)
            if code in un_downlimit:
                buy_available.append(code)

    return [buy_available,sell_available]

#获取停牌、收盘价、上市日期数据
susp_days=pd.read_csv('susp_days.csv',index_col=0)
close_price=pd.read_csv('close.csv',index_col=0)
time_now=datetime.now()
today=datetime.strftime(time_now,"%Y%m%d")
code=w.wset("sectorconstituent","date="+today+";sectorid=a001010100000000;field=wind_code").Data[0]
ipo=w.wss(code,"ipo_date")
code=map(lambda x:str(int(x[0:6])),code)
ipo_date=pd.Series(ipo.Data[0],index=code,name=ipo.Fields[0])

#通过收益率预测或者IC打分，结合可交易股票池,返回交易信号
# def stock_select(date,buy_available,sell_available):
#     buy_stock_list=[]
#     sell_stock_list=[]
#     return [buy_stock_list,sell_stock_list]


#买入函数,都按限价买入计算
def order_buy(date, stock_code, stocknum, order_price, deal_table, slippery, cost,cash_account,stock_account):

    #计算滑点、交易费用之后的成交价格,并入总的成交单表
    deal_price=round(order_price*(1+slippery) * (1+cost),2)
    deal_table=deal_table.append([date,stock_code,deal_price,stocknum])

    #计算剩余现金
    cash_account=cash_account-deal_price*stocknum

    #计算交易后的持仓
    stock_account[stock_code]=stock_account[stock_code]+stocknum

    return [deal_table,cash_account,stock_account]


#卖出函数，都按限价卖出计算
def order_sell(date, stock_code, stocknum, order_price, deal_table, slippery,cost,cash_account,stock_account):

    #卖出不考虑费用只考虑滑点,并入总的成交单表
    deal_price = round(order_price * (1 - slippery), 2)
    deal_table = deal_table.append([date,stock_code,deal_price,stocknum])

    # 计算剩余现金
    cash_account=cash_account+deal_price*stocknum

    # 计算交易后的持仓
    stock_account[stock_code] = stock_account[stock_code] + stocknum

    return [deal_table,cash_account,stock_account]

#计算截面账户净值
def my_account(date,cash_account,stock_account,close_price):
    market_value=0
    for stock in stock_account.index:
        stock_close_price=close_price.ix[date,stock]
        market_value+=stock_close_price*stock_account[stock]
    account_value=market_value + cash_account
    date_account=pd.Series([cash_account,market_value,account_value],index=['cash_account','market_value','account_value'])
    return date_account




