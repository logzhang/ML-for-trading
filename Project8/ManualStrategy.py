import numpy as np
import pandas as pd
import marketsimcode as msim
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from util import get_data
import indicators as ind
import datetime as dt

def author():
    return

def testPolicy(symbol, sd, ed, sv=100000):
    dates = pd.date_range(sd, ed)
    prices_all = get_data([symbol], dates)
    prices_all.fillna(method='ffill', inplace=True)
    prices_all.fillna(method='bfill', inplace=True)
    price = prices_all[symbol]
    price_norm = price / price.iloc[0]
    trades = price.copy() * 0
    holds = price.copy() * 0

    period = 14
    ema_ratio = ind.ind1_ema(price_norm, period)[1]
    momentum = ind.ind2_momentum(price_norm, period)
    bbp = ind.ind3_bbp(price_norm, period)[0]
    rsi = ind.ind4_rsi(price_norm, period)
    psy = ind.ind5_psy(price_norm, period)
    # std = ind.ind1_std(price_norm, period)

    for i in range(period-1, price_norm.shape[0]-1):
        # if ema_ratio.iloc[i]<0.95 and bbp.iloc[i]<0.1 and psy.iloc[i] < 30:
        if ema_ratio.iloc[i] < 0.95 and bbp.iloc[i] < 0.005 and rsi.iloc[i] < 30:
        # if ema_ratio.iloc[i] < 0.97 and bbp.iloc[i] < 0.0 and std.iloc[i] < 0.1:
            if holds.iloc[i-1] == -1000:
                trades.iloc[i] = 2000
            elif holds.iloc[i-1] == 0:
                trades.iloc[i] = 1000
            elif holds.iloc[i-1] == 1000:
                trades.iloc[i] = 0
            holds.iloc[i] = holds.iloc[i-1] + trades.iloc[i]
        # elif ema_ratio.iloc[i]>1.05 and bbp.iloc[i]>0.9 and psy.iloc[i] > 60:
        elif ema_ratio.iloc[i] > 1.05 and bbp.iloc[i] > 0.95 and rsi.iloc[i] > 55:
        # elif ema_ratio.iloc[i] > 1.03 and bbp.iloc[i] > 0.8 and std.iloc[i] < 0.1:
            if holds.iloc[i - 1] == -1000:
                trades.iloc[i] = 0
            elif holds.iloc[i - 1] == 0:
                trades.iloc[i] = -1000
            elif holds.iloc[i - 1] == 1000:
                trades.iloc[i] = -2000
            holds.iloc[i] = holds.iloc[i - 1] + trades.iloc[i]
        else:
            trades.iloc[i] = 0
            holds.iloc[i] = holds.iloc[i - 1] + trades.iloc[i]
    return trades.to_frame(name='buy_sell')

def create_chart(sd = dt.datetime(2008, 1, 1),ed = dt.datetime(2009, 12, 31), sv=100000, symbol = 'JPM',
                 chart_sd = pd.Timestamp('2007-12-01'), chart_ed = pd.Timestamp('2010-02-01'), chart_nm = 'p8_ms_in_sample'):
    opt_trade = testPolicy(symbol = symbol, sd=sd, ed=ed, sv = sv)
    # opt_trade.to_csv('manual_trade.csv')
    ben_trade = opt_trade.copy() * 0
    ben_trade.iloc[0] = 1000

    benchmark_portvals = msim.compute_portvals(orders=ben_trade, symbol = symbol, start_val=sv, commission=9.95, impact=0.005)
    benchmark_daily_rets = (benchmark_portvals / benchmark_portvals.shift(1)) - 1
    benchmark_daily_rets = benchmark_daily_rets[1:]
    benchmark_cr = (benchmark_portvals[-1] / benchmark_portvals[0] - 1)
    benchmark_adr = benchmark_daily_rets.mean()
    benchmark_sddr = benchmark_daily_rets.std()
    benchmark_sharpe_ratio = np.sqrt(252) * (benchmark_daily_rets).mean() / benchmark_daily_rets.std()
    benchmark_days = benchmark_portvals.shape[0]
    benchmark_portvals_normed = benchmark_portvals/benchmark_portvals.iloc[0]


    portfolio_portvals = msim.compute_portvals(orders=opt_trade, symbol = symbol, start_val=sv, commission=9.95, impact=0.005)
    portfolio_daily_rets = (portfolio_portvals / portfolio_portvals.shift(1)) - 1
    portfolio_daily_rets = portfolio_daily_rets[1:]
    portfolio_cr = (portfolio_portvals[-1] / portfolio_portvals[0] - 1)
    portfolio_adr = portfolio_daily_rets.mean()
    portfolio_sddr = portfolio_daily_rets.std()
    portfolio_sharpe_ratio = np.sqrt(252) * (portfolio_daily_rets).mean() / portfolio_daily_rets.std()
    portfolio_days = portfolio_portvals.shape[0]
    portfolio_portvals_normed = portfolio_portvals / portfolio_portvals.iloc[0]

    plt.figure(figsize=(14,7))
    plt.rc('font', size=18)
    plt.title('Manual Strategy vs. Benchmark')
    ax1 = benchmark_portvals_normed.plot(label='Benchmark', color = 'g')
    ax1 = portfolio_portvals_normed.plot(label='Manual_Strategy', color = 'r')

    ms_buy_index = opt_trade[opt_trade['buy_sell']>0].index
    # print (ms_buy_index)
    for d in (ms_buy_index):
        ax1.axvline(d, color='b')
    ms_sell_index = opt_trade[opt_trade['buy_sell'] < 0].index
    for d in (ms_sell_index):
        ax1.axvline(d, color='k')
    # print(ms_sell_index)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Normalized Asset Value')
    ax1.set_xlim(chart_sd, chart_ed)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=0)
    ax1.grid(True)
    ax1.legend()

    plt.savefig(chart_nm+'.png')
    metric_pd = pd.DataFrame({'Benchmark':[benchmark_cr,benchmark_sddr,benchmark_adr,benchmark_portvals[-1],
                                          benchmark_sharpe_ratio],
                             'Manual_Strategy':[portfolio_cr,portfolio_sddr,portfolio_adr,portfolio_portvals[-1],
                                          portfolio_sharpe_ratio]},
                             index = ['Cumulative return', 'STD of Daily Return','Mean of Daily Return',
                                      'Final Asset Value', 'Sharp Ratio'])
    return metric_pd

if __name__ == "__main__":
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    symbol = 'JPM'
    chart_sd = pd.Timestamp('2007-12-01')
    chart_ed = pd.Timestamp('2010-02-01')
    chart_nm = 'p8_ms_in_sample'
    ms_in_sample = create_chart(sd = sd, ed = ed, sv = sv, symbol = symbol,chart_sd= chart_sd, chart_ed =chart_ed,chart_nm = chart_nm)
    ms_in_sample.to_csv('p8_manual_metrics_in_sample.csv')

    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    sv = 100000
    symbol = 'JPM'
    chart_sd = pd.Timestamp('2009-12-01')
    chart_ed = pd.Timestamp('2012-02-01')
    chart_nm = 'p8_ms_out_sample'
    ms_out_sample = create_chart(sd=sd, ed=ed, sv=sv, symbol=symbol, chart_sd=chart_sd, chart_ed=chart_ed,chart_nm = chart_nm)
    ms_out_sample.to_csv('p8_manual_metrics_out_sample.csv')




