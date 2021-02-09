import indicators as ind
import TheoreticallyOptimalStrategy as tos
import datetime as dt
import pandas as pd

def test_project():
    #part 1 indicators
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    periodN = 14
    symbol = 'JPM'
    chart_sd = pd.Timestamp('2007-12-01')
    chart_ed = pd.Timestamp('2010-02-01')
    ind.test_chart(sd = sd, ed = ed, periodN = periodN, symbol = symbol,
    chart_sd = chart_sd, chart_ed = chart_ed)

    #part 2 theoritical optimal strategy
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    symbol = 'JPM'
    chart_sd = pd.Timestamp('2007-12-01')
    chart_ed = pd.Timestamp('2010-02-01')
    tos.create_chart(sd=sd, ed=ed, sv=sv, symbol=symbol, chart_sd=chart_sd, chart_ed=chart_ed)

def author():
    return
    
if __name__ == "__main__":
    test_project()
