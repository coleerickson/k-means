from parse_arff import Database
from random import sample

class KMeans:
    def __init__(self,k,database):
        self.database = database
        self.k = k


    def _learn(self):
        means = sample(self.database.data,k)
        while True:
            # my ginger
            clusters = [0 for _ in range(len(self.database.data))]

            # classify the eaxmples
            for i,example in enumerate(self.database.data):
                # Calculate which mean the point is closest to
                mean = 0 # do something
                clusters[i] = mean


            # recompute the kmean
            for cluster_num in range(k):
                cluster_examples = [ex for ex,c in zip(examples,clusters) if c == cluster_num]

                # how do you do the mean? lol




        pass

    def predict(self, example):
        pass
