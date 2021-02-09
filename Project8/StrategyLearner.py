""""""  		  	   		     		  		  		    	 		 		   		 		  
"""  		  	   		     		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
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
import random
import numpy as np
import pandas as pd  		  	   		     		  		  		    	 		 		   		 		  
import util as ut
import indicators as ind
import QLearner as ql
import marketsimcode as msim
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
class StrategyLearner(object):  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		     		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		     		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		     		  		  		    	 		 		   		 		  
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		     		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		     		  		  		    	 		 		   		 		  
    :param commission: The commission amount charged, defaults to 0.0  		  	   		     		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    # constructor  		  	   		     		  		  		    	 		 		   		 		  
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        Constructor method  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        self.verbose = verbose  		  	   		     		  		  		    	 		 		   		 		  
        self.impact = impact  		  	   		     		  		  		    	 		 		   		 		  
        self.commission = commission  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    # this method should create a QLearner, and train it for trading  		  	   		     		  		  		    	 		 		   		 		  
    def add_evidence(  		  	   		     		  		  		    	 		 		   		 		  
        self,  		  	   		     		  		  		    	 		 		   		 		  
        symbol="IBM",  		  	   		     		  		  		    	 		 		   		 		  
        sd=dt.datetime(2008, 1, 1),  		  	   		     		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 1, 1),  		  	   		     		  		  		    	 		 		   		 		  
        sv=100000,
    ):  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        Trains your strategy learner over a given time frame.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol to train on  		  	   		     		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		     		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		     		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		     		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		     		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		     		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		     		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
        # add your code to do learning here
        # example usage of the old backward compatible util function  		  	   		     		  		  		    	 		 		   		 		  
        syms = [symbol]  		  	   		     		  		  		    	 		 		   		 		  
        dates = pd.date_range(sd, ed)  		  	   		     		  		  		    	 		 		   		 		  
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices_all.fillna(method="ffill", inplace=True)
        prices_all.fillna(method="bfill", inplace=True)
        prices = prices_all[symbol]  # only portfolio symbols
        prices_SPY = prices_all["SPY"]  # only SPY, for comparison later
        prices_norm = prices/prices.iloc[0]
        prices_SPY_norm = prices_SPY/prices_SPY.iloc[0]
        trades = prices.copy() * 0
        holds = prices.copy() * 0
        Qreward = prices.copy() * 0

        if self.verbose:
            print(prices)

        period_N = 14
        ema = ind.ind1_ema(prices_norm, period_N)[1]
        bbp = ind.ind3_bbp(prices_norm, period_N)[0]
        rsi = ind.ind4_rsi(prices_norm, period_N)

        def Discretize_var(data, steps):
            sorted_data = np.sort(data)
            split_data = np.array_split(sorted_data, steps)
            cuts = [a[-1] for a in split_data]
            return np.digitize(data, cuts[:-1], right=True)
        disc_ema = Discretize_var(ema, 10)
        disc_bbp = Discretize_var(bbp, 10)
        disc_rsi = Discretize_var(rsi, 10)
        Qstate = disc_ema*100+disc_bbp*10+disc_rsi

        random.seed(2000)
        learner = ql.QLearner(
            num_states=1000,
            num_actions=3,
            alpha=0.2,
            gamma=0.9,
            rar=0.5,
            radr=0.99,
            dyna=0,
            verbose=False,
        )  # initialize the learner
        daily_return = prices_norm/prices_norm.shift(1) - 1
        daily_return.iloc[0] = 0
        trades_iter = trades.copy() + 10
        # action = learner.querysetstate(Qstate[period_N - 2])
        count = 0
        while (trades.equals(trades_iter) == False) & (count < 10000):
            trades_iter = trades.copy()
            #Compute the current state(including holding) Qstate.ilot[i]
            #Compute the reward for the last action Qreward.iloc[i-1]
            #Query the learner with the current state and reward to get an action
            # action = 0, buy active = 1, sell action = 2, nothing
            #Implement the action the learner returned (LONG, CASH, SHORT), and update portfolio value
            for i in range(period_N-1, prices.shape[0]-1):
                action = learner.query(Qstate[i], Qreward.iloc[i - 1])
                if action == 0:
                    if holds.iloc[i - 1] == -1000:
                        trades.iloc[i] = 2000
                    elif holds.iloc[i - 1] == 0:
                        trades.iloc[i] = 1000
                    elif holds.iloc[i - 1] == 1000:
                        trades.iloc[i] = 0
                    holds.iloc[i] = holds.iloc[i - 1] + trades.iloc[i]
                elif action == 1:
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

                if trades.iloc[i] == 0:
                    Qreward.iloc[i] = holds.iloc[i] * daily_return.iloc[i + 1] * prices.iloc[i]
                else:
                    Qreward.iloc[i] = holds.iloc[i] * daily_return.iloc[i + 1] * prices.iloc[i] - abs(
                        trades.iloc[i] * prices[i] * self.impact) - self.commission
            holds.iloc[-1] = holds.iloc[-2]
            trades.iloc[-1] = 0
            count = count + 1
        if count == 10000:
            print("timeout")
        df_trades = trades.to_frame(name='buy_sell')
        self.learner = learner
        return df_trades

    # this method should use the existing policy and test it against new data  		  	   		     		  		  		    	 		 		   		 		  
    def testPolicy(  		  	   		     		  		  		    	 		 		   		 		  
        self,  		  	   		     		  		  		    	 		 		   		 		  
        symbol="IBM",  		  	   		     		  		  		    	 		 		   		 		  
        sd=dt.datetime(2010, 1, 1),
        ed=dt.datetime(2011, 12, 31),
        sv=100000,
    ):  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        Tests your learner using data outside of the training data  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol that you trained on on  		  	   		     		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		     		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		     		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		     		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		     		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		     		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		     		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		     		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		     		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		     		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		     		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		     		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		  	   		     		  		  		    	 		 		   		 		  
        """
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices_all.fillna(method="ffill", inplace=True)
        prices_all.fillna(method="bfill", inplace=True)
        prices = prices_all[symbol]  # only portfolio symbols
        prices_SPY = prices_all["SPY"]  # only SPY, for comparison later
        prices_norm = prices / prices.iloc[0]
        prices_SPY_norm = prices_SPY / prices_SPY.iloc[0]
        trades = prices.copy() * 0
        holds = prices.copy() * 0
        Qreward = prices.copy() * 0

        period_N = 14
        ema = ind.ind1_ema(prices_norm, period_N)[1]
        bbp = ind.ind3_bbp(prices_norm, period_N)[0]
        rsi = ind.ind4_rsi(prices_norm, period_N)

        def Discretize_var(data, steps):
            sorted_data = np.sort(data)
            split_data = np.array_split(sorted_data, steps)
            cuts = [a[-1] for a in split_data]
            return np.digitize(data, cuts[:-1], right=True)

        disc_ema = Discretize_var(ema, 10)
        disc_bbp = Discretize_var(bbp, 10)
        disc_rsi = Discretize_var(rsi, 10)
        Qstate = disc_ema * 100 + disc_bbp * 10 + disc_rsi

        # random.seed(1000)
        learner = self.learner
        # Implement the action the learner returned (LONG, CASH, SHORT), and update portfolio value
        # action = learner.querysetstate(Qstate[period_N-2])
        for i in range(period_N - 1, prices.shape[0]-1):
            action = learner.querysetstate(Qstate[i])
            if action == 0:
                if holds.iloc[i - 1] == -1000:
                    trades.iloc[i] = 2000
                elif holds.iloc[i - 1] == 0:
                    trades.iloc[i] = 1000
                elif holds.iloc[i - 1] == 1000:
                    trades.iloc[i] = 0
                holds.iloc[i] = holds.iloc[i - 1] + trades.iloc[i]
            elif action == 1:
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
        holds.iloc[-1] = holds.iloc[-2]
        trades.iloc[-1] = 0
        df_trades = trades.to_frame(name='buy_sell')
        return df_trades

    def author():
        return

if __name__ == "__main__":  		  	   		     		  		  		    	 		 		   		 		  
    print("One does not simply think up a strategy")
