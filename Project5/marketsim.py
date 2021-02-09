""""""  		  	   		     		  		  		    	 		 		   		 		  
"""MC2-P1: Market simulator.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		     		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		     		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		     		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		     		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		     		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		     		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		     		  		  		    	 		 		   		 		  
or edited.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		     		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		     		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		     		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  		  	   		     		  		  		    	 		 		   		 		  
"""  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		     		  		  		    	 		 		   		 		  
import os  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
import numpy as np  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		     		  		  		    	 		 		   		 		  
from util import get_data, plot_data  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
def compute_portvals(  		  	   		     		  		  		    	 		 		   		 		  
    orders_file="./orders/orders.csv",  		  	   		     		  		  		    	 		 		   		 		  
    start_val=1000000,  		  	   		     		  		  		    	 		 		   		 		  
    commission=9.95,  		  	   		     		  		  		    	 		 		   		 		  
    impact=0.005,  		  	   		     		  		  		    	 		 		   		 		  
):  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    Computes the portfolio values.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    :param orders_file: Path of the order file or the file object  		  	   		     		  		  		    	 		 		   		 		  
    :type orders_file: str or file object  		  	   		     		  		  		    	 		 		   		 		  
    :param start_val: The starting value of the portfolio  		  	   		     		  		  		    	 		 		   		 		  
    :type start_val: int  		  	   		     		  		  		    	 		 		   		 		  
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		     		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		     		  		  		    	 		 		   		 		  
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		     		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		     		  		  		    	 		 		   		 		  
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		     		  		  		    	 		 		   		 		  
    :rtype: pandas.DataFrame  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    # this is the function the autograder will call to test your code  		  	   		     		  		  		    	 		 		   		 		  
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		     		  		  		    	 		 		   		 		  
    # code should work correctly with either input  		  	   		     		  		  		    	 		 		   		 		  
    # TODO: Your code here
    #read in order files
    orders = pd.read_csv(orders_file, parse_dates=True, na_values=["nan"])
    orders.sort_values(by = 'Date', inplace = True)
    orders.reset_index(inplace=True, drop = True)
    #get order start date, end date and stock list
    start_date = orders['Date'][0]
    end_date = orders['Date'][orders.shape[0]-1]
    symbols = list(set(orders['Symbol']))
    #use get_data to read in stock price
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(symbols, dates, addSPY=True, colname="Adj Close")
    prices_all.fillna(method = 'ffill', inplace=True)
    prices_all.fillna(method = 'bfill', inplace=True)
    prices_all.rename(columns = {'SPY': 'CASH'}, inplace=True)
    prices_all['CASH'] = 1.0
    #Create a df trades changes
    trades = prices_all.copy()
    columns = trades.columns
    trades[columns] = 0.0
    for row in orders.index:
        if orders.loc[row,'Order'] == 'BUY':
            trades.loc[orders.loc[row, 'Date'], orders.loc[row, 'Symbol']] += orders.loc[row,'Shares']
        else:
            trades.loc[orders.loc[row, 'Date'], orders.loc[row, 'Symbol']] -= orders.loc[row, 'Shares']
        trades.loc[orders.loc[row, 'Date'], 'CASH'] += -1*commission - abs(orders.loc[row, 'Shares'] * prices_all.loc[orders.loc[row, 'Date'], orders.loc[row, 'Symbol']] * impact)
    for symbol in symbols:
        trades['CASH'] += trades[symbol] * prices_all[symbol] * (-1)
    #create a df holding
    holding = trades.copy()
    holding.iloc[0]['CASH'] = holding.iloc[0]['CASH'] + start_val
    holding_f = holding.cumsum(axis=0)
    #create a df values = prices * holding
    value = pd.DataFrame(holding_f.values * prices_all.values, columns=holding_f.columns, index=holding_f.index)
    #create a df total value
    portvals = value.sum(axis=1)
    return portvals

def author():
    return 'ngao33'

def test_code():  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    Helper function to test code  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    # this is a helper function you can use to test your code  		  	   		     		  		  		    	 		 		   		 		  
    # note that during autograding his function will not be called.  		  	   		     		  		  		    	 		 		   		 		  
    # Define input parameters  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    # of = "./marketsim/orders/orders-11.csv"
    # of = "./marketsim/additional_orders/orders-short.csv"
    of = "./marketsim/orders/test.csv"
    sv = 1000000
    cm = 9.95
    impc = 0.005
  		  	   		     		  		  		    	 		 		   		 		  
    # Process orders  		  	   		     		  		  		    	 		 		   		 		  
    portvals = compute_portvals(orders_file=of, start_val=sv, commission=cm, impact=impc)
    # GET SPY
    sd = portvals.index[0]
    ed = portvals.index[-1]
    dates = pd.date_range(sd, ed)
    prices_all = get_data(["$SPX"], dates)  # automatically adds SPY
    prices = prices_all["$SPX"]  # only portfolio symbols
    spx_daily_rets = (prices / prices.shift(1)) - 1
    spx_daily_rets = spx_daily_rets[1:]
    spx_cr = (prices[-1] / prices[0] - 1)
    spx_adr = spx_daily_rets.mean()
    spx_sddr = spx_daily_rets.std()
    spx_sharpe_ratio = np.sqrt(252) * (spx_daily_rets).mean() / spx_daily_rets.std()

    if isinstance(portvals, pd.DataFrame):  		  	   		     		  		  		    	 		 		   		 		  
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		     		  		  		    	 		 		   		 		  
    else:  		  	   		     		  		  		    	 		 		   		 		  
        "warning, code did not return a DataFrame"  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    # Get portfolio stats  		  	   		     		  		  		    	 		 		   		 		  
    # Here we just fake the data. you should use your code from previous assignments.
    daily_rets = (portvals / portvals.shift(1)) - 1
    daily_rets = daily_rets[1:]
    cr = (portvals[-1] / portvals[0] - 1)
    adr = daily_rets.mean()
    sddr = daily_rets.std()
    sharpe_ratio = np.sqrt(252) * (daily_rets).mean() / daily_rets.std()
    days = portvals.shape[0]

    # Compare portfolio against $SPX  		  	   		     		  		  		    	 		 		   		 		  
    print(f"How many days: {days}")
    print(f"Cumulative Return of Fund: {portvals[days-1]}")
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Average Daily Return of Fund: {adr}")
    print(f"Standard Deviation of Fund: {sddr}")
    print()

    print(f"Date Range: {sd} to {ed}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {spx_sharpe_ratio}")
    print()
    print(f"Cumulative Return of Fund: {cr}")
    print(f"Cumulative Return of SPY : {spx_cr}")
    print()
    print(f"Standard Deviation of Fund: {sddr}")
    print(f"Standard Deviation of SPY : {spx_sddr}")
    print()
    print(f"Average Daily Return of Fund: {adr}")
    print(f"Average Daily Return of SPY : {spx_adr}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")

if __name__ == "__main__":  		  	   		     		  		  		    	 		 		   		 		  
    test_code()  		  	   		     		  		  		    	 		 		   		 		  
