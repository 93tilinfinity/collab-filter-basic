"""
3 classes - UserCF, ItemCF, simpleCF

simpleCF - inherits MLutils, MLmetrics
UserCf, ItemCF - standalone
"""
from MLutils import MovieLens
from MLutils import EvaluationData
import MLmetrics

from surprise import KNNBasic
from collections import defaultdict
import heapq
import time

def getSimsMatrix(trainset,name,user_based = False):
    # Build a similarity matrix using surprise lib's KNNBasic module
    sim_options = {'name':name,'user_based':user_based}
    model = KNNBasic(sim_options=sim_options)
    model.fit(trainset)
    return model.sim

def scorefunc(rating,SimScore):
    return rating * SimScore

# USER BASED CF
class UserCF():
    def __init__(self, trainSet, trainSet_looxv, testSet_looxv,similarity):
        self.trainSet, self.trainSet_looxv, self.testSet_looxv = trainSet, trainSet_looxv, testSet_looxv
        self.similarity = similarity

        t0 = time.time()
        self.simsMatrix_looxv = getSimsMatrix(trainSet_looxv, similarity, True)
        t1 = time.time()
        self.topN = self.predict()
        t2 = time.time()
        print('User Matrix:',t1-t0,'User TopN:',t2-t1)

    def predict(self,n=8):
        # Build Top List Based On X-Validated Training Data
        topN = defaultdict(list)
        k = 20
        for uiid in range(self.trainSet_looxv.n_users):
            # Get k most similar users
            similarityRow = self.simsMatrix_looxv[uiid]
            similarUsers = []
            for innerID,score in enumerate(similarityRow):
                if innerID != uiid:
                    similarUsers.append((innerID,score))
            kNeighbours = heapq.nlargest(k,similarUsers,key=lambda x:x[1])

            # Build Candidate List
            candidates = defaultdict(float)
            candidates_weight = defaultdict(float)
            for innerID,userSimScore in kNeighbours:
                otherRatings = self.trainSet_looxv.ur[innerID]
                for rating in otherRatings:
                    candidates[rating[0]] += scorefunc(rating[1],userSimScore)
                    candidates_weight[rating[0]] += userSimScore

            # Build Seen List
            hasWatched = [item[0] for item in self.trainSet_looxv.ur[uiid]]

            # Produce TopN from Candidates dict less hasWatched
            pos = 0
            for itemID,score in sorted(candidates.items(),key=lambda x:x[1],reverse = True):
                if not hasWatched.__contains__(int(itemID)):
                    movieID = self.trainSet_looxv.to_raw_iid(itemID)
                    topN[int(self.trainSet_looxv.to_raw_uid(uiid))].append((int(movieID),score))
                    # print(ml.getMovieName(int(movieID)),score/candidates_weight[itemID])
                    pos +=1
                    if pos > n:
                        break
        return topN

# ITEM BASED CF
class ItemCF():
    def __init__(self, trainSet, trainSet_looxv, testSet_looxv,similarity,size):
        self.trainSet, self.trainSet_looxv, self.testSet_looxv = trainSet, trainSet_looxv, testSet_looxv
        self.similarity = similarity
        self.modelsize = size

        t0 = time.time()
        self.simsMatrix_looxv = getSimsMatrix(trainSet_looxv, similarity, False)
        t1 = time.time()
        self.topN = self.predict()
        t2 = time.time()
        print('Item Matrix:',t1-t0,'Item TopN:',t2-t1)

    def predict(self,n=8):
        # Build Top List Based On X-Validated Training Data
        topN = defaultdict(list)
        k = 20
        for uiid in range(self.trainSet_looxv.n_users):
            # Get k top rated items
            activeUserRatings = self.trainSet_looxv.ur[uiid]
            kNeighbours = heapq.nlargest(k, activeUserRatings, lambda x: x[1])

            # Build Candidate List
            candidates = defaultdict(float)
            candidates_weight = defaultdict(float)

            for iiid, rating in kNeighbours:
                # for each top item rated, get all similar items.
                similarityRow = self.simsMatrix_looxv[iiid]
                if self.modelsize:
                    # Implement 'model size' Truncation.
                    similarityRow = heapq.nlargest(self.modelsize, similarityRow)
                for other_iiid, itemSimScore in enumerate(similarityRow):
                    candidates[other_iiid] += scorefunc(rating, itemSimScore)
                    candidates_weight[other_iiid] += itemSimScore

            # Build Seen List
            hasWatched = [item[0] for item in self.trainSet_looxv.ur[uiid]]

            # Produce TopN from Candidates dict less hasWatched
            pos = 0
            for itemID, score in sorted(candidates.items(), key=lambda x: x[1], reverse=True):
                if not hasWatched.__contains__(int(itemID)):
                    movieID = self.trainSet_looxv.to_raw_iid(itemID)
                    topN[int(self.trainSet_looxv.to_raw_uid(uiid))].append((int(movieID), score))
                    # print(ml.getMovieName(int(movieID)),score/candidates_weight[itemID])
                    pos += 1
                    if pos > n:
                        break
        return topN

class simpleCF():
    def __init__(self, MovieLensObject):
        self.ml = MovieLensObject
        self.data = ml.load1Mdata()
        self.trainSet, self.trainSet_looxv, self.testSet_looxv = self.processData(self.data)
        self.testUserInnerID = self.testUserSummary('56')

    def runUserCF(self,similarity):
        usercf = UserCF(self.trainSet, self.trainSet_looxv, self.testSet_looxv, similarity)
        return usercf.predict()

    def runItemCF(self,similarity,modelsize=None):
        itemcf = ItemCF(self.trainSet, self.trainSet_looxv, self.testSet_looxv, similarity,modelsize)
        return itemcf.predict()

    def processData(self,data):
        print('preparing data...')
        eval = EvaluationData(data)
        return eval.trainSet, eval.LOOX_trainSet, eval.LOOX_testSet

    def testUserSummary(self,testUser):
        testUserInnerID = self.trainSet.to_inner_uid(testUser)
        print("Target User Total Ratings:", len(self.trainSet.ur[testUserInnerID]))
        print("Target User 5 Star Ratings:")
        for iid, rating in self.trainSet.ur[testUserInnerID]:
            if rating == 5.0:
                print(ml.getMovieName(int(self.trainSet.to_raw_iid(iid))))
        return testUserInnerID

    def RecommenderMetrics(self,topN,testUser):
        print('-hitRate:',MLmetrics.HitRate(topN,self.testSet_looxv))
        print('-ratinghitRate:',MLmetrics.RatingHitRate(topN,self.testSet_looxv))
        # print('-cumulativehitRate:',MLmetrics.CumulativeHitRate(topN,self.testSet_looxv))
        # print('-AverageReciprocalhitRank:',MLmetrics.ARHR(topN,self.testSet_looxv))

        print('-Target User Top 8 List:')
        counter = 0
        for movieID,score in topN[int(testUser)]:
            print(ml.getMovieName(int(movieID)))
            counter += 1
            if counter > 7:
                break

t = time.time()
ml = MovieLens()
rankings = ml.getPopularityRanking()
cf = simpleCF(ml)
print('Load data:',time.time() - t)

user = cf.runUserCF('cosine')

# pearson = cf.runItemCF('pearson')
# cosine = cf.runItemCF('cosine')
# msd = cf.runItemCF('msd')

modelsize10 = cf.runItemCF('cosine',modelsize = 10)
# modelsize30 = cf.runItemCF('cosine',modelsize = 30)
# modelsize50 = cf.runItemCF('cosine',modelsize = 50)

cf.RecommenderMetrics(modelsize10,'56')

cf.RecommenderMetrics(user,'56')