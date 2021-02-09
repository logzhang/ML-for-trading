import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data

def compute_portvals(
        orders,
        symbol,
        start_val=1000000,
        commission=0.00,
        impact=0.00,
):
    order = orders.copy()
    order.sort_index(inplace=True)
    order.reset_index(inplace=True, drop=False)
    order.rename(columns ={'index':'Date'}, inplace=True)
    # get order start date, end date and stock list
    start_date = order['Date'][0]
    end_date = order['Date'][order.shape[0] - 1]
    order['Symbol'] = symbol
    order['Shares'] = abs(order['buy_sell'])
    symbols = list(set(order['Symbol']))
    # use get_data to read in stock price
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(symbols, dates, addSPY=True, colname="Adj Close")
    prices_all.fillna(method='ffill', inplace=True)
    prices_all.fillna(method='bfill', inplace=True)
    if symbol == 'SPY':
        prices_all['CASH'] = prices_all['SPY']
    else:
        prices_all.rename(columns={'SPY': 'CASH'}, inplace=True)
    prices_all['CASH'] = 1.0
    # Create a df trades changes
    trades = prices_all.copy()
    columns = trades.columns
    trades[columns] = 0.0
    for row in order.index:
        if order.loc[row, 'buy_sell'] > 0:
            trades.loc[order.loc[row, 'Date'], order.loc[row, 'Symbol']] += order.loc[row, 'Shares']
            trades.loc[order.loc[row, 'Date'], 'CASH'] += -1 * commission - abs(
                order.loc[row, 'Shares'] * prices_all.loc[order.loc[row, 'Date'], order.loc[row, 'Symbol']] * impact)
        elif order.loc[row, 'buy_sell'] < 0:
            trades.loc[order.loc[row, 'Date'], order.loc[row, 'Symbol']] -= order.loc[row, 'Shares']
            trades.loc[order.loc[row, 'Date'], 'CASH'] += -1 * commission - abs(
                order.loc[row, 'Shares'] * prices_all.loc[order.loc[row, 'Date'], order.loc[row, 'Symbol']] * impact)
    for symbol in symbols:
        trades['CASH'] += trades[symbol] * prices_all[symbol] * (-1)
    # create a df holding
    holding = trades.copy()
    holding.iloc[0]['CASH'] = holding.iloc[0]['CASH'] + start_val
    holding_f = holding.cumsum(axis=0)
    # create a df values = prices * holding
    value = pd.DataFrame(holding_f.values * prices_all.values, columns=holding_f.columns, index=holding_f.index)
    # create a df total value
    portvals = value.sum(axis=1)
    return portvals

def author():
    return

def test_code():
    pass

if __name__ == "__main__":
    test_code()