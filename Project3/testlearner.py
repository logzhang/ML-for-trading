""""""  		  	   		     		  		  		    	 		 		   		 		  
"""  		  	   		     		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
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
  		  	   		     		  		  		    	 		 		   		 		  
import math  		  	   		     		  		  		    	 		 		   		 		  
import sys  		  	   		     		  		  		    	 		 		   		 		  
import time
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
  		  	   		     		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		     		  		  		    	 		 		   		 		  
    if len(sys.argv) != 2:  		  	   		     		  		  		    	 		 		   		 		  
        print("Usage: python testlearner.py <filename>")  		  	   		     		  		  		    	 		 		   		 		  
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.genfromtxt(inf, delimiter=",")
    data = data[1:, 1:]

    # compute how much of the data is training and testing  		  	   		     		  		  		    	 		 		   		 		  
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    train_x = data[:train_rows, 0:-1]  		  	   		     		  		  		    	 		 		   		 		  
    train_y = data[:train_rows, -1]  		  	   		     		  		  		    	 		 		   		 		  
    test_x = data[train_rows:, 0:-1]  		  	   		     		  		  		    	 		 		   		 		  
    test_y = data[train_rows:, -1]  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    # print(f"{test_x.shape}")
    # print(f"{test_y.shape}")
    # experiment 1
    rmse_in_sample = np.zeros([50])
    rmse_out_sample = np.zeros([50])
    for i in range(1,51):
        learner = dt.DTLearner(leaf_size=i, verbose=False)  # constructor
        learner.add_evidence(train_x, train_y)  # training step
        # evaluate in sample
        pred_y = learner.query(train_x)  # get the predictions
        rmse_in_sample[i-1] = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        # evaluate out of sample
        pred_y = learner.query(test_x)  # get the predictions
        rmse_out_sample[i-1] = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    pd.DataFrame(rmse_in_sample).to_csv('experiment1_rmse_in_sample.csv')
    pd.DataFrame(rmse_out_sample).to_csv('experiment1_rmse_out_sample.csv')

    plt.figure(1)
    plt.xlabel('leaf_size')
    plt.ylabel('RMSE')
    plt.title('P3_Figure1')
    x = np.arange(1, 51)
    plt.plot(x, rmse_in_sample, Marker = 'o', markersize=3, label = 'RMSE in Sample')
    plt.plot(x, rmse_out_sample, Marker = 'o',  markersize=3, label = 'RMSE out Sample')
    plt.legend(loc=0)
    plt.savefig('P3_Figure1.png')

    # experiment 2
    bags = [10,20,50]
    rmse_in_sample = np.zeros([len(bags),50])
    rmse_out_sample = np.zeros([len(bags),50])
    for bag in range(len(bags)):
        for i in range(1, 51):
            learner = bl.BagLearner(learner = dt.DTLearner, kwargs = {"leaf_size":i}, bags = bags[bag], boost = False, verbose = False)
            learner.add_evidence(train_x, train_y)  # training step
            # evaluate in sample
            pred_y = learner.query(train_x)  # get the predictions
            rmse_in_sample[bag, i - 1] = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
            # evaluate out of sample
            pred_y = learner.query(test_x)  # get the predictions
            rmse_out_sample[bag, i - 1] = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    # print(rmse_in_sample)
    # print(rmse_out_sample)
    plt.figure(2)
    plt.xlabel('leaf_size')
    plt.ylabel('RMSE')
    plt.title('P3_Figure2')
    x = np.arange(1, 51)
    plt.plot(x, rmse_in_sample[0], color='b', label='RMSE in Sample bag 10')
    plt.plot(x, rmse_out_sample[0],linestyle='dashed', color='b', label='RMSE out Sample bag 10')
    plt.plot(x, rmse_in_sample[1], color='g',label='RMSE in Sample bag 20')
    plt.plot(x, rmse_out_sample[1],linestyle='dashed', color='g', label='RMSE out Sample bag 20')
    plt.plot(x, rmse_in_sample[2], color='r', label='RMSE in Sample bag 50')
    plt.plot(x, rmse_out_sample[2], linestyle='dashed', color='r', label='RMSE out Sample bag 50')
    plt.legend(loc=0)
    plt.savefig('P3_Figure2.png')

    # experiment 3
    # measurement 1 runtime
    # measurement 2 mean absolute error
    # measurement 3 R^2
    np.random.seed(202009)
    np.random.shuffle(data)
    DT_train_time = np.zeros([50])
    DT_query_time = np.zeros([50])
    DT_test_MAE = np.zeros([50])
    DT_train_RS = np.zeros([50])
    DT_test_RS = np.zeros([50])
    DT_train_RMSE = np.zeros([50])
    DT_test_RMSE = np.zeros([50])
    for i in range(1,51):
        learner = dt.DTLearner(leaf_size=i, verbose=False)  # constructor
        start_time = time.time()
        learner.add_evidence(train_x, train_y)  # training step
        DT_train_time[i-1] = time.time()-start_time
        # evaluate in sample
        start_time = time.time()
        pred_y = learner.query(train_x)  # get the predictions
        DT_query_time[i-1] = time.time() - start_time
        DT_train_RS[i-1] =1 - ((train_y - pred_y) ** 2).sum() / ((train_y - train_y.mean()) ** 2).sum()
        DT_train_RMSE[i - 1] = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        # evaluate out of sample
        pred_y = learner.query(test_x)  # get the predictions
        DT_test_MAE[i - 1] = ((abs(test_y - pred_y)).sum() / test_y.shape[0])
        DT_test_RS[i - 1] = 1 - ((test_y - pred_y) ** 2).sum() / ((test_y - test_y.mean()) ** 2).sum()
        DT_test_RMSE[i - 1] = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    RT_train_time = np.zeros([50])
    RT_query_time = np.zeros([50])
    RT_test_MAE = np.zeros([50])
    RT_train_RS = np.zeros([50])
    RT_test_RS = np.zeros([50])
    RT_train_RMSE = np.zeros([50])
    RT_test_RMSE = np.zeros([50])
    for i in range(1, 51):
        learner = rt.RTLearner(leaf_size=i, verbose=False)  # constructor
        start_time = time.time()
        learner.add_evidence(train_x, train_y)  # training step
        RT_train_time[i-1] = time.time() - start_time
        # evaluate in sample
        start_time = time.time()
        pred_y = learner.query(train_x)  # get the predictions
        RT_query_time[i-1] = time.time() - start_time
        RT_train_RS[i - 1] = 1 - ((train_y - pred_y) ** 2).sum() / ((train_y - train_y.mean()) ** 2).sum()
        RT_train_RMSE[i - 1] = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        # evaluate out of sample
        pred_y = learner.query(test_x)  # get the predictions
        RT_test_MAE[i - 1] = ((abs(test_y - pred_y)).sum() / test_y.shape[0])
        RT_test_RS[i - 1] = 1 - ((test_y - pred_y) ** 2).sum() / ((test_y - test_y.mean()) ** 2).sum()
        RT_test_RMSE[i - 1] = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    plt.figure(3)
    plt.xlabel('leaf_size')
    plt.ylabel('Model Training Time')
    plt.title('Model Training Time')
    x = np.arange(1, 51)
    plt.plot(x, DT_train_time, Marker = 'o', markersize=3, label = 'DT Learner')
    plt.plot(x, RT_train_time, Marker = 'o',  markersize=3, label = 'RT Learner')
    plt.legend(loc=0)
    plt.savefig('P3_Figure3.png')

    plt.figure(4)
    plt.xlabel('leaf_size')
    plt.ylabel('Model MAE')
    plt.title('Out-of-sample MAE')
    x = np.arange(1, 51)
    plt.ylim(0, 0.008)
    plt.plot(x, DT_test_MAE, Marker = 'o', markersize=3, label = 'DT Learner')
    plt.plot(x, RT_test_MAE, Marker = 'o',  markersize=3, label = 'RT Learner')
    plt.legend(loc=0)
    plt.savefig('P3_Figure4.png')

    plt.figure(5)
    plt.xlabel('leaf_size')
    plt.ylabel('Out-of-Sample R^2')
    plt.title('Out-of-Sample R^2')
    x = np.arange(1, 51)
    plt.ylim(0, 0.8)
    plt.plot(x, DT_test_RS, Marker = 'o', markersize=3, label = 'DT Learner')
    plt.plot(x, RT_test_RS, Marker = 'o',  markersize=3, label = 'RT Learner')
    plt.legend(loc=0)
    plt.savefig('P3_Figure5.png')

    plt.figure(6)
    plt.xlabel('leaf_size')
    plt.ylabel('In-Sample R^2')
    plt.title('P3_Figure6')
    x = np.arange(1, 51)
    plt.plot(x, DT_train_RS, Marker = 'o', markersize=3, label = 'DT Learner')
    plt.plot(x, RT_train_RS, Marker = 'o',  markersize=3, label = 'RT Learner')
    plt.legend(loc=0)
    plt.savefig('P3_Figure6.png')

    plt.figure(7)
    plt.xlabel('leaf_size')
    plt.ylabel('Out-of-Sample RMSE')
    plt.title('P3_Figure7')
    x = np.arange(1, 51)
    plt.ylim(0, 0.01)
    plt.plot(x, DT_test_RMSE, Marker = 'o', markersize=3, label = 'DT Learner')
    plt.plot(x, RT_test_RMSE, Marker = 'o',  markersize=3, label = 'RT Learner')
    plt.legend(loc=0)
    plt.savefig('P3_Figure7.png')

    plt.figure(8)
    plt.xlabel('leaf_size')
    plt.ylabel('In-Sample RMSE')
    plt.title('P3_Figure8')
    x = np.arange(1, 51)
    plt.ylim(0, 0.01)
    plt.plot(x, DT_train_RMSE, Marker = 'o', markersize=3, label = 'DT Learner')
    plt.plot(x, RT_train_RMSE, Marker = 'o',  markersize=3, label = 'RT Learner')
    plt.legend(loc=0)
    plt.savefig('P3_Figure8.png')
