import numpy as np
import pandas as pd
import marketsimcode as msim
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ManualStrategy as ms
import StrategyLearner as sl
import datetime as dt

def author():
    return

def exp2(sd = dt.datetime(2008, 1, 1),ed = dt.datetime(2009, 12, 31), sv=100000, symbol = 'JPM',
                 chart_sd = pd.Timestamp('2007-12-01'), chart_ed = pd.Timestamp('2010-02-01'),
                 impact = 0.00, commission = 0.00):

    learner = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)
    learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    sl_trade = learner.testPolicy(symbol=symbol, \
                              sd=sd, \
                              ed=ed, \
                              sv=sv)
    sl_portvals = msim.compute_portvals(orders=sl_trade, symbol=symbol, start_val=sv, commission=commission, impact=impact)
    sl_daily_rets = (sl_portvals / sl_portvals.shift(1)) - 1
    sl_daily_rets = sl_daily_rets[1:]
    sl_cr = (sl_portvals[-1] / sl_portvals[0] - 1)
    sl_adr = sl_daily_rets.mean()
    sl_sddr = sl_daily_rets.std()
    sl_sharpe_ratio = np.sqrt(252) * (sl_daily_rets).mean() / sl_daily_rets.std()
    sl_days = sl_portvals.shape[0]
    sl_portvals_normed = sl_portvals / sl_portvals.iloc[0]
    s1_action_days = sl_trade[sl_trade['buy_sell'] != 0].shape[0]

    metric_pd = pd.DataFrame({'Strategy_Learner': [sl_cr, sl_sddr, sl_adr, sl_portvals[-1],
                                                  sl_sharpe_ratio, s1_action_days]
                              },
                             index = ['Cumulative return', 'STD of Daily Return','Mean of Daily Return',
                                      'Final Asset Value', 'Sharp Ratio','Actions'])
    return metric_pd, sl_portvals_normed

if __name__ == "__main__":
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    symbol = 'JPM'
    chart_sd = pd.Timestamp('2007-12-01')
    chart_ed = pd.Timestamp('2010-02-01')
    impact = [0.00, 0.005, 0.010, 0.02]
    commission = 0.00
    metric_pd =  exp2(sd=sd, ed=ed, sv=sv, symbol=symbol, chart_sd=chart_sd, chart_ed=chart_ed, impact = impact[0],
         commission=commission)[0].rename(columns={'Strategy_Learner': 0})
    for i in range(1,len(impact)):
        temp = exp2(sd=sd, ed=ed, sv=sv, symbol=symbol, chart_sd=chart_sd, chart_ed=chart_ed, impact = impact[i],
         commission=commission)[0].rename(columns={'Strategy_Learner': i})
        metric_pd = metric_pd.join(temp)
    metric_pd.to_csv('p8_experiment2.csv')

    plt.figure(figsize=(14,7))
    plt.rc('font', size=18)
    plt.title('Strategy Learner Impact Value')
    for i in range(len(impact)):
        temp = exp2(sd=sd, ed=ed, sv=sv, symbol=symbol, chart_sd=chart_sd, chart_ed=chart_ed, impact=impact[i],
                    commission=commission)[1]
        ax1 = temp.plot(label='Impact'+str(impact[i]))
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Normalized Asset Value')
    ax1.set_xlim(chart_sd, chart_ed)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=0)
    ax1.grid(True)
    ax1.legend()

    plt.savefig('p8_experiment2.png')
