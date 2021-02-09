""""""  		  	   		     		  		  		    	 		 		   		 		  
"""  		  	   		     		  		  		    	 		 		   		 		  
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
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
  		  	   		     		  		  		    	 		 		   		 		  
import numpy as np  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
class RTLearner(object):
    """
    This is a Linear Regression Learner. It is implemented correctly.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """
    def __init__(self, leaf_size = 1, verbose=False):
        """
        Constructor method
        """
        self.leaf_size = leaf_size
        self.verbose = verbose
        if self.verbose == True:
            print ('tree')
        pass  # move along, these aren't the drones you're looking for

    def author(self):
        """  		  	   		     		  		  		    	 		 		   		 		  
        :return: The GT username of the student  		  	   		     		  		  		    	 		 		   		 		  
        :rtype: str  		  	   		     		  		  		    	 		 		   		 		  
        """
        return  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        """  		  	   		     		  		  		    	 		 		   		 		  
        Add training data to learner  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
        :param data_x: A set of feature values used to train the learner  		  	   		     		  		  		    	 		 		   		 		  
        :type data_x: numpy.ndarray  		  	   		     		  		  		    	 		 		   		 		  
        :param data_y: The value we are attempting to predict given the X data  		  	   		     		  		  		    	 		 		   		 		  
        :type data_y: numpy.ndarray  		  	   		     		  		  		    	 		 		   		 		  
        """
        data_y_t = np.array([data_y]).T
        data = np.append(data_x, data_y_t, axis=1)
        # build and save the model
        def build_tree(data):
            if data.shape[0] <= self.leaf_size:
                return ['leaf', data[:,-1].mean(), 'NA', 'NA']
            elif np.all(data[:, -1] == data[0, -1]):
                return ['leaf', data[0, -1], 'NA', 'NA']
            else:
                i = np.random.randint(data.shape[1] - 1)
                # SplitVal = (data[np.random.randint(data.shape[0]),i] + data[np.random.randint(data.shape[0]),i])/2.0
                SplitVal = np.median(data[:, i])
                if data[data[:, i] <= SplitVal].shape[0] == 0 or data[data[:, i] > SplitVal].shape[0] == 0:
                    return ['leaf', data[:, -1].mean(), 'NA', 'NA']
                else:
                    lefttree = build_tree(data[data[:, i] <= SplitVal])
                    righttree = build_tree(data[data[:, i] > SplitVal])
                    if np.array(lefttree).ndim == 1:
                        root = [i, SplitVal, 1, 2]
                    else:
                        root = [i, SplitVal, 1, np.array(lefttree).shape[0] + 1]
                    return np.vstack((root, lefttree, righttree))
        self.model = build_tree(data)
    def query(self, points):  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        Estimate a set of test points given the model we built.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		     		  		  		    	 		 		   		 		  
        :type points: numpy.ndarray  		  	   		     		  		  		    	 		 		   		 		  
        :return: The predicted result of the input data according to the trained model  		  	   		     		  		  		    	 		 		   		 		  
        :rtype: numpy.ndarray  		  	   		     		  		  		    	 		 		   		 		  
        """
        result = np.ones([points.shape[0],1])
        for i in range(points.shape[0]):
            row = 0
            while self.model[row][0] != 'leaf':
                if points[i][int(float(self.model[row][0]))] <= float(self.model[row][1]):
                    row = int(row+float(self.model[row][2]))
                else:
                    row = int(row + float(self.model[row][3]))
            result[i][0] = float(self.model[row][1])
        return result[:,-1]
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		     		  		  		    	 		 		   		 		  
    print("the secret clue is 'zzyzx'")  		  	   		     		  		  		    	 		 		   		 		  
