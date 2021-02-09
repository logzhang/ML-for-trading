import numpy as np
import BagLearner as bl
import LinRegLearner as lrl
class InsaneLearner(object):
    def __init__(self, verbose = False):
        self.verbose = verbose
    def author(self):
        return
    def add_evidence(self, data_x, data_y):
        self.model = np.array([])
        for i in range(20):
            self.model = np.append(self.model, bl.BagLearner(learner = lrl.LinRegLearner, kwargs = {}, bags = 20, boost = False, verbose = self.verbose))
            self.model[i].add_evidence(data_x, data_y)
    def query(self, points):
        result_insane = np.ones([points.shape[0], 20])
        for i in range(20):
            result_insane[:,i] = self.model[i].query(points)
        return result_insane.mean(axis = 1)
if __name__ == "__main__":  		  	   		     		  		  		    	 		 		   		 		  
    print("the secret clue is 'zzyzx'")  		  	   		     		  		  		    	 		 		   		 		  
