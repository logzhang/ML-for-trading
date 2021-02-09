

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from util import get_data

#EMA
def ind1_ema(price, period):
    sma = price.rolling(window=period, min_periods=period).mean()[:period]
    rest = price[period:]
    ema = pd.concat([sma, rest]).ewm(span = period, adjust = False).mean()
    price_ema_ratio = price / ema
    return ema, price_ema_ratio

#Momentum
def ind2_momentum(price, period):
    momentum = price / price.shift(period) - 1
    return momentum

#Bollinger bands
def ind3_bbp(price, period):
    std = price.rolling(window = period, min_periods= period).std()
    sma = price.rolling(window = period, min_periods= period).mean()
    bb_upper = sma + 2*std
    bb_lower = sma - 2*std
    bbp = (price - bb_lower)/(bb_upper - bb_lower)
    return bbp, bb_upper, bb_lower

#Relative strength index (RSI)
def ind4_rsi(price, period):
    daily_return = price - price.shift(1)
    tot_gain = daily_return.copy()
    tot_gain[tot_gain < 0] = 0
    avg_gain = tot_gain.rolling(window=period, min_periods=period).sum() / period
    avg_gain_emv = avg_gain.copy()
    for i in range(period+1, avg_gain.shape[0]):
        avg_gain_emv.iloc[i] = (avg_gain_emv.iloc[i-1]*(period-1) + tot_gain.iloc[i])/period
    tot_loss = daily_return.copy()
    tot_loss[tot_loss > 0] = 0
    avg_loss = -1 * tot_loss.rolling(window=period, min_periods=period).sum() / period
    avg_loss_emv = avg_loss.copy()
    for i in range(period+1, avg_loss.shape[0]):
        avg_loss_emv.iloc[i] = (avg_loss_emv.iloc[i - 1] * (period-1) - tot_loss.iloc[i]) / period
    rs = avg_gain_emv / avg_loss_emv
    rsi = 100 - (100 / (1 + rs))
    return rsi
#Psychological Line (PSY)
def ind5_psy(price, period):
    daily_return = price - price.shift(1)
    daily_return[daily_return>0] = 1
    daily_return[daily_return<=0] = 0
    psy = daily_return.rolling(window = period, min_periods= period).mean()*100
    return psy

def author():
    return

def test_chart(sd = '2008-01-01', ed = '2009-12-31', periodN = 14, symbol = 'JPM',
               chart_sd = pd.Timestamp('2007-02-01'), chart_ed = pd.Timestamp('2010-02-01')
               ):
    dates = pd.date_range(sd, ed)
    prices_all = get_data([symbol], dates)
    prices_all.fillna(method='ffill', inplace=True)
    prices_all.fillna(method='bfill', inplace=True)
    price = prices_all[symbol]
    # price.to_csv('jpm_prices.csv')
    price_norm = price/price.iloc[0]

    # volume_all = get_data(["JPM"], dates, colname="Volume")
    # volume = volume_all['JPM']
    # print (prices_all.head())
    # print (type(price))
    # print (type(prices_all))
    # print(type(price_norm))

    #EMA indicator chart
    ema = ind1_ema(price_norm, periodN)[0]
    price_ema = ind1_ema(price_norm, periodN)[1]
    # ema.to_frame().to_csv('ema14.csv')
    # price_ema.to_frame().to_csv('price_ema14.csv')
    # print (ema)
    # print (price_ema)
    plt.figure(figsize=(16,8))
    plt.rc('font', size=18)
    plt.subplot(2,1,1)
    plt.title('Exponential Moving Average')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    ax1 = ema.plot(label = 'EMA%i' %periodN)
    ax1 = price_norm.plot(label = 'Normalized Adj. Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Normalized Price')
    ax1.set_xlim(chart_sd, chart_ed)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=0)
    ax1.grid(True)
    ax1.legend()

    plt.subplot(2,1,2)
    ax2 = price_ema.plot(label = 'Price/EMA%i Ratio' %periodN, color = 'g')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price/EMA Ratio')
    ax2.set_xlim(chart_sd, chart_ed)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=0)
    ax2.grid(True)
    ax2.legend()
    plt.savefig('ind1_ema')

    #Momentum chart
    momem = ind2_momentum(price_norm, periodN)
    # momem.to_frame().to_csv('momem14.csv')
    plt.figure(figsize=(16,8))
    plt.rc('font', size=18)
    plt.subplot(2,1,1)
    plt.title('Momentum')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    ax1 = momem.plot(label = 'Momentum%i' %periodN)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Momentum')
    ax1.set_xlim(chart_sd, chart_ed)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=0)
    ax1.grid(True)
    ax1.legend()

    plt.subplot(2,1,2)
    ax2 = price_norm.plot(label = 'Normalized Adj. Price', color = 'g')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Normalized Price')
    ax2.set_xlim(chart_sd, chart_ed)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=0)
    ax2.grid(True)
    ax2.legend()
    plt.savefig('ind2_momentum')

    #BBP indicator chart
    bbp = ind3_bbp(price_norm, periodN)[0]
    bb_upper = ind3_bbp(price_norm, periodN)[1]
    bb_lower = ind3_bbp(price_norm, periodN)[2]
    # bbp.to_frame().to_csv('bbp14.csv')
    # bb_upper.to_frame().to_csv('bb_upper.csv')
    # bb_lower.to_frame().to_csv('bb_lower.csv')
    plt.figure(figsize=(18,9))
    plt.rc('font', size=18)
    plt.subplot(2,1,1)
    plt.title('Bollinger Bands')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    ax1 = bb_upper.plot(label = 'Upper Bands')
    ax1 = bb_lower.plot(label = 'Lower Bands')
    ax1 = price_norm.plot(label = 'Normalized Adj. Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Normalized Price')
    ax1.set_xlim(chart_sd, chart_ed)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=0)
    ax1.grid(True)
    ax1.legend()

    plt.subplot(2,1,2)
    ax2 = bbp.plot(label = 'Bollinger Bands %', color = 'r')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Bollinger Bands %')
    ax2.set_xlim(chart_sd, chart_ed)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=0)
    ax2.grid(True)
    ax2.legend()
    plt.savefig('ind3_bbp')

    #RSI chart
    rsi = ind4_rsi(price_norm, periodN)
    # rsi.to_frame().to_csv('rsi14.csv')
    plt.figure(figsize=(16,8))
    plt.rc('font', size=18)
    plt.subplot(2,1,1)
    plt.title('Relative Strength Index')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    ax1 = rsi.plot(label = 'RSI%i' %periodN)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('RSI')
    ax1.set_xlim(chart_sd, chart_ed)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=0)
    ax1.grid(True)
    ax1.legend()
    plt.axhline(y=30, color = 'r')
    plt.axhline(y=70, color = 'r')

    plt.subplot(2,1,2)
    ax2 = price_norm.plot(label = 'Normalized Adj. Price', color = 'g')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Normalized Price')
    ax2.set_xlim(chart_sd, chart_ed)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=0)
    ax2.grid(True)
    ax2.legend()
    plt.savefig('ind4_rsi')

    #PSY chart
    psy = ind5_psy(price_norm, periodN)
    # psy.to_frame().to_csv('psy14.csv')
    plt.figure(figsize=(16, 8))
    plt.rc('font', size=18)
    plt.subplot(2, 1, 1)
    plt.title('Psychological Line')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    ax1 = psy.plot(label='PSY%i' %periodN)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('PSY')
    ax1.set_xlim(chart_sd, chart_ed)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=0)
    ax1.grid(True)
    ax1.legend()
    plt.axhline(y=30, color='r')
    plt.axhline(y=70, color='r')

    plt.subplot(2, 1, 2)
    ax2 = price_norm.plot(label='Normalized Adj. Price', color='g')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Normalized Price')
    ax2.set_xlim(chart_sd, chart_ed)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=0)
    ax2.grid(True)
    ax2.legend()
    plt.savefig('ind5_psy')

if __name__ == "__main__":
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    periodN = 14
    symbol = 'JPM'
    chart_sd = pd.Timestamp('2007-12-01')
    chart_ed = pd.Timestamp('2010-02-01')
    test_chart(sd = sd, ed = ed, periodN = periodN, symbol = symbol,
    chart_sd = chart_sd, chart_ed = chart_ed)

