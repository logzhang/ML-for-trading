""""""  		  	   		     		  		  		    	 		 		   		 		  
"""Assess a betting strategy.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
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
  		  	   		     		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		     		  		  		    	 		 		   		 		  
GT User ID: tb34 (replace with your User ID)  		  	   		     		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		  	   		     		  		  		    	 		 		   		 		  
"""  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
import numpy as np
import matplotlib.pyplot as plt


def author():  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		     		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    return "xzhang947"  # replace tb34 with your Georgia Tech username.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
def gtid():  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    :return: The GT ID of the student  		  	   		     		  		  		    	 		 		   		 		  
    :rtype: int  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    return  # replace with your GT ID number  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
def get_spin_result(win_prob):  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    :param win_prob: The probability of winning  		  	   		     		  		  		    	 		 		   		 		  
    :type win_prob: float  		  	   		     		  		  		    	 		 		   		 		  
    :return: The result of the spin.  		  	   		     		  		  		    	 		 		   		 		  
    :rtype: bool  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    result = False  		  	   		     		  		  		    	 		 		   		 		  
    if np.random.random() <= win_prob:  		  	   		     		  		  		    	 		 		   		 		  
        result = True  		  	   		     		  		  		    	 		 		   		 		  
    return result  		  	   		     		  		  		    	 		 		   		 		  

def bet_strategy(win_prob):
    episode_winnings = 0
    bets = 1
    winnings = np.zeros(1001)
    # print('winnings', winnings)
    while episode_winnings < 80 and bets < 1001:
        won = False
        bet_amount = 1
        while not won:
            won = get_spin_result(win_prob)
            if won == True:
                episode_winnings = episode_winnings + bet_amount
            else:
                episode_winnings = episode_winnings - bet_amount
                bet_amount = bet_amount * 2
            winnings[bets] = episode_winnings
            bets += 1
    if episode_winnings >= 80:
        winnings[bets:] = 80
    print('winning', winnings)
    return winnings

def bet_strategy_realistic(win_prob):
    total_money = 256
    episode_winnings = 0
    bets = 1
    winnings = np.zeros(1001)
    while episode_winnings < 80 and episode_winnings > -1 * total_money and bets < 1001:
        won = False
        bet_amount = 1
        while not won:
            won = get_spin_result(win_prob)
            if won == True:
                episode_winnings = episode_winnings + bet_amount
            else:
                episode_winnings = episode_winnings - bet_amount
                bet_amount = min(bet_amount * 2, episode_winnings + total_money)
            winnings[bets] = episode_winnings
            bets += 1
    if episode_winnings >= 80:
        winnings[bets:] = 80
    elif episode_winnings <= -1 * total_money:
        winnings[bets:] = -1 * total_money
    return winnings

def test_code():  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    Method to test your code  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    win_prob = 18.0/38  # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once  		  	   		     		  		  		    	 		 		   		 		  
    # print(get_spin_result(win_prob))  # test the roulette spin
    # add your code here to implement the experiments
    # plot code reference
    # https://jakevdp.github.io/PythonDataScienceHandbook/04.01-simple-line-plots.html
    # Experiment 1
    run_times = 10
    exp1 = np.zeros((run_times,1001))
    for i in range(run_times):
        exp1[i] = bet_strategy(win_prob)
    plt.figure(1)
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.xlabel('Bets')
    plt.ylabel('Winnings')
    plt.title('Figure1')
    x = np.arange(0, 1001)
    for i in range(0, run_times):
        plt.plot(x, exp1[i])
    # plt.savefig('Figure1.png')
    plt.show()

    # Experiment 2
    run_times = 1000
    exp2 = np.zeros((run_times,1001))
    for i in range(run_times):
        exp2[i] = bet_strategy(win_prob)
    plt.figure(2)
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.xlabel('Bets')
    plt.ylabel('Winnings')
    plt.title('Figure2')
    x = np.arange(0, 1001)
    mean = exp2.mean(axis=0)
    mean_above = exp2.mean(axis=0) + exp2.std(axis = 0)
    mean_below = exp2.mean(axis=0) - exp2.std(axis=0)
    plt.plot(x, mean, label = 'mean')
    plt.plot(x, mean_above, label = 'mean + sd')
    plt.plot(x, mean_below, label = 'mean - sd')
    plt.legend(loc=0)
    # plt.savefig('Figure2.png')
    plt.show()

    # Experiment 3
    run_times = 1000
    exp3 = exp2
    plt.figure(3)
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.xlabel('Bets')
    plt.ylabel('Winnings')
    plt.title('Figure3')
    x = np.arange(0, 1001)
    median = np.median(exp3, axis=0)
    median_above = np.median(exp3, axis=0) + exp3.std(axis = 0)
    median_below = np.median(exp3, axis=0) - exp3.std(axis=0)
    plt.plot(x, median, label = 'median')
    plt.plot(x, median_above, label = 'median + sd')
    plt.plot(x, median_below, label = 'median - sd')
    plt.legend(loc=0)
    # plt.savefig('Figure3.png')
    plt.show()

    # Experiment 4
    run_times = 1000
    exp4 = np.zeros((run_times,1001))
    for i in range(run_times):
        exp4[i] = bet_strategy_realistic(win_prob)
    plt.figure(4)
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.xlabel('Bets')
    plt.ylabel('Winnings')
    plt.title('Figure4')
    x = np.arange(0, 1001)
    mean = np.mean(exp4, axis=0)
    mean_above = np.mean(exp4, axis=0) + exp4.std(axis = 0)
    mean_below = np.mean(exp4, axis=0) - exp4.std(axis = 0)
    plt.plot(x, mean, label = 'mean')
    plt.plot(x, mean_above, label = 'mean + sd')
    plt.plot(x, mean_below, label = 'mean - sd')
    plt.legend(loc=0)
    # plt.savefig('Figure4.png')
    plt.show()

    # Experiment 5
    run_times = 1000
    exp5 = exp4
    plt.figure(5)
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.xlabel('Bets')
    plt.ylabel('Winnings')
    plt.title('Figure5')
    x = np.arange(0, 1001)
    median = np.median(exp5, axis=0)
    median_above = np.median(exp5, axis=0) + exp5.std(axis=0)
    median_below = np.median(exp5, axis=0) - exp5.std(axis=0)
    plt.plot(x, median, label = 'median')
    plt.plot(x, median_above, label = 'median + sd')
    plt.plot(x, median_below, label = 'median - sd')
    plt.legend(loc=0)
    # plt.savefig('Figure5.png')
    plt.show()

if __name__ == "__main__":  		  	   		     		  		  		    	 		 		   		 		  
    test_code()  		  	   		     		  		  		    	 		 		   		 		  
