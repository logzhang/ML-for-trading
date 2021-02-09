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

def exp1(sd = dt.datetime(2008, 1, 1),ed = dt.datetime(2009, 12, 31), sv=100000, symbol = 'JPM',
                 chart_sd = pd.Timestamp('2007-12-01'), chart_ed = pd.Timestamp('2010-02-01'),
                 impact = 0.005, commission = 9.95):
    opt_trade = ms.testPolicy(symbol = symbol, sd=sd, ed=ed, sv = sv)
    # opt_trade.to_csv('manual_trade.csv')
    ben_trade = opt_trade.copy() * 0
    ben_trade.iloc[0] = 1000

    benchmark_portvals = msim.compute_portvals(orders=ben_trade, symbol = symbol, start_val=sv, commission=commission, impact=impact)
    benchmark_daily_rets = (benchmark_portvals / benchmark_portvals.shift(1)) - 1
    benchmark_daily_rets = benchmark_daily_rets[1:]
    benchmark_cr = (benchmark_portvals[-1] / benchmark_portvals[0] - 1)
    benchmark_adr = benchmark_daily_rets.mean()
    benchmark_sddr = benchmark_daily_rets.std()
    benchmark_sharpe_ratio = np.sqrt(252) * (benchmark_daily_rets).mean() / benchmark_daily_rets.std()
    benchmark_days = benchmark_portvals.shape[0]
    benchmark_portvals_normed = benchmark_portvals/benchmark_portvals.iloc[0]

    portfolio_portvals = msim.compute_portvals(orders=opt_trade, symbol = symbol, start_val=sv, commission=commission, impact=impact)
    portfolio_daily_rets = (portfolio_portvals / portfolio_portvals.shift(1)) - 1
    portfolio_daily_rets = portfolio_daily_rets[1:]
    portfolio_cr = (portfolio_portvals[-1] / portfolio_portvals[0] - 1)
    portfolio_adr = portfolio_daily_rets.mean()
    portfolio_sddr = portfolio_daily_rets.std()
    portfolio_sharpe_ratio = np.sqrt(252) * (portfolio_daily_rets).mean() / portfolio_daily_rets.std()
    portfolio_days = portfolio_portvals.shape[0]
    portfolio_portvals_normed = portfolio_portvals / portfolio_portvals.iloc[0]

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

    plt.figure(figsize=(14,7))
    plt.rc('font', size=18)
    plt.title('Strategy Learner vs. Manual Strategy vs. Benchmark')
    ax1 = benchmark_portvals_normed.plot(label='Benchmark', color = 'g')
    ax1 = portfolio_portvals_normed.plot(label='Manual_Strategy', color = 'r')
    ax1 = sl_portvals_normed.plot(label='Strategy_Learner', color='b')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Normalized Asset Value')
    ax1.set_xlim(chart_sd, chart_ed)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=0)
    ax1.grid(True)
    ax1.legend()

    plt.savefig('p8_experiment1.png')
    metric_pd = pd.DataFrame({'Benchmark':[benchmark_cr,benchmark_sddr,benchmark_adr,benchmark_portvals[-1],
                                          benchmark_sharpe_ratio],
                             'Manual_Strategy':[portfolio_cr,portfolio_sddr,portfolio_adr,portfolio_portvals[-1],
                                          portfolio_sharpe_ratio],
                              'Strategy_Learner': [sl_cr, sl_sddr, sl_adr, sl_portvals[-1],
                                                  sl_sharpe_ratio]
                              },
                             index = ['Cumulative return', 'STD of Daily Return','Mean of Daily Return',
                                      'Final Asset Value', 'Sharp Ratio'])
    metric_pd.to_csv('p8_experiment1.csv')

if __name__ == "__main__":
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    symbol = 'JPM'
    chart_sd = pd.Timestamp('2007-12-01')
    chart_ed = pd.Timestamp('2010-02-01')
    impact = 0.005
    commission = 9.95
    exp1(sd=sd, ed=ed, sv=sv, symbol=symbol, chart_sd=chart_sd, chart_ed=chart_ed, impact = impact,
         commission=commission)
