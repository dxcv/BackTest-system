import pandas as pd
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt


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

a=math.sqrt(2)
for i in range(1000000):
    a=math.sqrt(2)**a
    print(a)