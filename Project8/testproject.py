import numpy as np
import pandas as pd
import marketsimcode as msim
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ManualStrategy as ms
import StrategyLearner as sl
import experiment1 as e1
import experiment2 as e2
import datetime as dt

def test_project():
    #part 1 Manual Strategy
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    symbol = 'JPM'
    chart_sd = pd.Timestamp('2007-12-01')
    chart_ed = pd.Timestamp('2010-02-01')
    chart_nm = 'p8_ms_in_sample'
    ms_in_sample = ms.create_chart(sd=sd, ed=ed, sv=sv, symbol=symbol, chart_sd=chart_sd, chart_ed=chart_ed,
                                chart_nm=chart_nm)
    ms_in_sample.to_csv('p8_manual_metrics_in_sample.csv')

    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    sv = 100000
    symbol = 'JPM'
    chart_sd = pd.Timestamp('2009-12-01')
    chart_ed = pd.Timestamp('2012-02-01')
    chart_nm = 'p8_ms_out_sample'
    ms_out_sample = ms.create_chart(sd=sd, ed=ed, sv=sv, symbol=symbol, chart_sd=chart_sd, chart_ed=chart_ed,
                                 chart_nm=chart_nm)
    ms_out_sample.to_csv('p8_manual_metrics_out_sample.csv')

    # part 2 Experiment 1
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    symbol = 'JPM'
    chart_sd = pd.Timestamp('2007-12-01')
    chart_ed = pd.Timestamp('2010-02-01')
    impact = 0.005
    commission = 9.95
    e1.exp1(sd=sd, ed=ed, sv=sv, symbol=symbol, chart_sd=chart_sd, chart_ed=chart_ed, impact = impact,
         commission=commission)

    # part 3 Experiment 2
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    symbol = 'JPM'
    chart_sd = pd.Timestamp('2007-12-01')
    chart_ed = pd.Timestamp('2010-02-01')
    impact = [0.00, 0.005, 0.010, 0.02]
    commission = 0.00
    metric_pd = e2.exp2(sd=sd, ed=ed, sv=sv, symbol=symbol, chart_sd=chart_sd, chart_ed=chart_ed, impact=impact[0],
                     commission=commission)[0].rename(columns={'Strategy_Learner': 0})
    for i in range(1, len(impact)):
        temp = e2.exp2(sd=sd, ed=ed, sv=sv, symbol=symbol, chart_sd=chart_sd, chart_ed=chart_ed, impact=impact[i],
                    commission=commission)[0].rename(columns={'Strategy_Learner': i})
        metric_pd = metric_pd.join(temp)
    metric_pd.to_csv('p8_experiment2.csv')

    plt.figure(figsize=(14, 7))
    plt.rc('font', size=18)
    plt.title('Strategy Learner Impact Value')
    for i in range(len(impact)):
        temp = e2.exp2(sd=sd, ed=ed, sv=sv, symbol=symbol, chart_sd=chart_sd, chart_ed=chart_ed, impact=impact[i],
                    commission=commission)[1]
        ax1 = temp.plot(label='Impact' + str(impact[i]))
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Normalized Asset Value')
    ax1.set_xlim(chart_sd, chart_ed)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=0)
    ax1.grid(True)
    ax1.legend()
    plt.savefig('p8_experiment2.png')

def author():
    return

if __name__ == "__main__":
    test_project()
