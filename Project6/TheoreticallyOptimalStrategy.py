# import TheoreticallyOptimalStrategy as tos
# df_trades = tos.testPolicy(symbol = "AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000)
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import marketsimcode as msim
import matplotlib.dates as mdates
from util import get_data

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

    #initialize trades and holds
    if price_norm.iloc[1] > price_norm.iloc[0]:
        holds[0] = 1000
        trades[0] = 1000
    elif price_norm.iloc[1] < price_norm.iloc[0]:
        holds[0] = -1000
        trades[0] = -1000
    else:
        holds[0] = 0
        trades[0] = 0
    #calculate holds and trades
    for i in range (1, price_norm.shape[0] - 1):
        if price_norm.iloc[i+1] > price_norm.iloc[i]:
            if holds[i-1] == -1000:
                trades[i] = 2000
            elif holds[i-1] == 0:
                trades[i] = 1000
            elif holds[i-1] == 1000:
                trades[i] = 0
            holds[i] = holds[i-1] + trades[i]
        elif price_norm.iloc[i+1] < price_norm.iloc[i]:
            if holds[i-1] == -1000:
                trades[i] = 0
            elif holds[i-1] == 0:
                trades[i] = -1000
            elif holds[i-1] == 1000:
                trades[i] = -2000
            holds[i] = holds[i-1] + trades[i]
        else:
            trades[i] = 0
            holds[i] = holds[i-1] + trades[i]
    return trades.to_frame(name = 'buy_sell')

def create_chart(sd = dt.datetime(2008, 1, 1),ed = dt.datetime(2009, 12, 31), sv=100000, symbol = 'JPM',
                 chart_sd = pd.Timestamp('2007-12-01'), chart_ed = pd.Timestamp('2010-02-01')):
    opt_trade = testPolicy(symbol = symbol, sd=sd, ed=ed, sv = sv)
    ben_trade = opt_trade.copy() * 0
    ben_trade.iloc[0] = 1000

    benchmark_portvals = msim.compute_portvals(orders=ben_trade, symbol = symbol, start_val=sv, commission=0.00, impact=0.00)
    benchmark_daily_rets = (benchmark_portvals / benchmark_portvals.shift(1)) - 1
    benchmark_daily_rets = benchmark_daily_rets[1:]
    benchmark_cr = (benchmark_portvals[-1] / benchmark_portvals[0] - 1)
    benchmark_adr = benchmark_daily_rets.mean()
    benchmark_sddr = benchmark_daily_rets.std()
    benchmark_sharpe_ratio = np.sqrt(252) * (benchmark_daily_rets).mean() / benchmark_daily_rets.std()
    benchmark_days = benchmark_portvals.shape[0]
    benchmark_portvals_normed = benchmark_portvals/benchmark_portvals.iloc[0]

    portfolio_portvals = msim.compute_portvals(orders=opt_trade, symbol = symbol, start_val=sv, commission=0.00, impact=0.00)
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
    plt.title('Optimal Strategy vs. Benchmark')
    ax1 = benchmark_portvals_normed.plot(label='Benchmark', color = 'g')
    ax1 = portfolio_portvals_normed.plot(label='Optimal', color = 'r')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Normalized Asset Value')
    ax1.set_xlim(chart_sd, chart_ed)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=0)
    ax1.grid(True)
    ax1.legend()

    plt.savefig('p6_optimal.png')
    metric_pd = pd.DataFrame({'Benchmark':[benchmark_cr,benchmark_sddr,benchmark_adr,benchmark_portvals[-1],
                                          benchmark_sharpe_ratio],
                             'Portfolio':[portfolio_cr,portfolio_sddr,portfolio_adr,portfolio_portvals[-1],
                                          portfolio_sharpe_ratio]},
                             index = ['Cumulative return', 'STD of Daily Return','Mean of Daily Return',
                                      'Final Asset Value', 'Sharp Ratio'])
    metric_pd.to_csv('p6_part2_metrics.csv')

if __name__ == "__main__":
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    symbol = 'JPM'
    chart_sd = pd.Timestamp('2007-12-01')
    chart_ed = pd.Timestamp('2010-02-01')
    create_chart(sd = sd, ed = ed, sv = sv, symbol = symbol,chart_sd= chart_sd, chart_ed =chart_ed)







