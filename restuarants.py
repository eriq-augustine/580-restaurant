import math
import sys
import random

import nltk

import parser
import features

def crossValidate(documents, classifierCreator = lambda trainSet: NBClassifier(trainSet),  numFolds = 4, seed = None):
   if seed:
      random.seed(seed)

   ordering = [i for i in range(0, len(documents))]
   # TEST
   #random.shuffle(ordering)

   folds = []
   for i in range(0, numFolds):
      folds.append([])

   count = 0
   for index in ordering:
      folds[count % numFolds].append(documents[index])
      count += 1

   rmse = 0
   for i in range(0, numFolds):
      testSet = folds[i]
      trainSet = []

      for j in range(0, numFolds):
         if i != j:
            trainSet += folds[j]

      classy = classifierCreator(trainSet)
      predictions = []
      for testDocument in testSet:
         # TEST
         predictions.append(classy.classifyDocument(testDocument))
         #predictions.append(classy.classifyDocument(testDocument[0]))
      foldRmse = calcRmse(predictions, [doc[1] for doc in testSet])
      rmse += foldRmse

      #TEST
      print foldRmse
      #print [document[1] for document in documents]
      #print predictions


   return rmse / numFolds

# Note that the inputs are numeric vecotrs
def calcRmse(actual, predicted):
   mse = 0

   for i in range(0, len(actual)):
      mse += math.pow(actual[i] - predicted[i], 2)

   return math.sqrt(float(mse) / len(actual))

class NBClassifier:
   def __init__(self, trainingSet, fsg = features.FeatureSetGenerator()):
      self.fsg = fsg
      self.fsg.defineAllFeatures([ doc[0] for doc in trainingSet ])

      #self.classy = nltk.DecisionTreeClassifier.train(self.labeledDocsToFeatures(trainingSet))

      self.classy = nltk.NaiveBayesClassifier.train(self.labeledDocsToFeatures(trainingSet), nltk.probability.ELEProbDist)
      #self.classy = nltk.NaiveBayesClassifier.train(self.labeledDocsToFeatures(trainingSet), nltk.probability.LaplaceProbDist)
      ##self.classy = nltk.NaiveBayesClassifier.train(self.labeledDocsToFeatures(trainingSet), nltk.probability.MLEProbDist)
      ##self.classy = nltk.NaiveBayesClassifier.train(self.labeledDocsToFeatures(trainingSet), nltk.probability.GoodTuringProbDist)

   def classifyDocument(self, document):
      return self.classy.classify(self.fsg.toFeatures(document))

   def classifyFeatures(self, features):
      return self.classy.classify(features)

   def classifier(self):
      return self.classy

   def accuracy(self, testSet):
      return nltk.classify.accuracy(self.classy, self.labeledDocsToFeatures(testSet))

   def showInfo(self):
      #TEST
      #print self.classy.pp()
      self.classy.show_most_informative_features(40)

   # documents = [ ( document, class ) ]
   # return = [ ( {features}, class ) ]
   def labeledDocsToFeatures(self, documents):
      return [(self.fsg.toFeatures(doc[0]), doc[1]) for doc in documents]

   def unlabeledDocsToFeatures(self, documents):
      return [self.fsg.toFeatures(doc[0]) for doc in documents]

class WordWeightClassifier:
   def __init__(self, trainingSet, minOccur = 6, maxOccurPercent = 0.20):
      self.weights = {}
      self.meanScore = 0

      maxOccur = int(len(trainingSet) * maxOccurPercent)

      rawScores = {}
      for trainDocument in trainingSet:
         unigrams = set(features.batchStem(features.toUnigrams(trainDocument[0])))
         self.meanScore += int(trainDocument[1])

         for gram in unigrams:
            if not rawScores.has_key(gram):
               rawScores[gram] = []
            rawScores[gram].append(int(trainDocument[1]))

      self.meanScore /= float(len(trainingSet))
      # TEST
      print 'Prior Prob: {0}'.format(self.meanScore)

      #compressedScores = {}
      for (gram, scores) in rawScores.items():
         if len(scores) >= minOccur and len(scores) < maxOccur:
            #compressedScores[gram] = scores
            self.weights[gram] = sum(scores) / float(len(scores))

      #for (key, val) in compressedScores.items():
      #   print '{0} -- {1}'.format(key, val)

      #for (key, val) in self.weights.items():
      #   print '{0} -- {1}'.format(key, val)

   def classifyDocument(self, document):
      #TEST
      #unigrams = features.batchStem(features.toUnigrams(document))
      unigrams = features.batchStem(features.toUnigrams(document[0]))

      scores = []
      mean = 0
      count = 0

      for gram in unigrams:
         if self.weights.has_key(gram):
            scores.append(self.weights[gram])
            mean += self.weights[gram]

      # No counting words here, assign the prior probability
      if len(scores) == 0:
         return self.meanScore
      mean /= len(scores)

      positives = []
      negatives = []

      # Get me some residuals, yum yum.
      squareResiduals = []
      for score in scores:
         #if score < mean:
         if score < self.meanScore:
            negatives.append(self.meanScore - score)
            #negatives.append(mean - score)
            #negatives.append(score)
         else:
            positives.append(score - self.meanScore)
            #positives.append(score - mean)
            #positives.append(score)

         sign = 1
         if score < mean:
            sign = -1
         squareResiduals.append(sign * math.pow(mean - score, 2))

      squareSum = sum(squareResiduals)

      sign = 1
      if squareSum < 0:
         sign = -1

      crazyScore = mean + sign * math.sqrt(abs(squareSum))

      #positiveScore = sum(positives)
      #negativeScore = sum(negatives)
      if len(positives) == 0:
         positiveScore = 0.0005
      else:
         positiveScore = sum(positives)
         #positiveScore = sum(positives) * len(scores) / len(positives)

      if len(negatives) == 0:
         negativeScore = 0.0005
      else:
         negativeScore = sum(negatives)
         #negativeScore = sum(negatives) * len(scores) / len(negatives)

      evenScore = positiveScore / negativeScore * mean

      sortedScores = sorted(scores)
      median = sortedScores[int(len(sortedScores) / 2)]

      print 'Even -- Positive: {0}, Negative: {1}, Even Score: {2}'.format(positiveScore, negativeScore, evenScore)
      print 'Mean: {0}, Median: {1}, CrazyScore: {2}, Real Score: {3}'.format(mean, median, crazyScore, document[1])

      rtn = 4
      if positiveScore > negativeScore:
         #rtn = int(0.5 + median)
         rtn = int(0.5 + mean)
      else:
         #rtn = int(median)
         #rtn = int(median - 0.5)
         rtn = int(mean - 0.5)

      print 'Final Score: {0}'.format(rtn)

      return rtn
      #return median
      #return self.meanScore
      #return 3.8
      #return crazyScore

def extractReviews(reviews):
   rtn = []
   for review in reviews:
      for cat in ['food', 'service', 'venue', 'overall']:
         rtn.append((review[cat + 'Review'], int(review[cat + 'Score'])))
   return rtn

if __name__ == '__main__':
   reviews = parser.readAllReviews()
   tests = parser.readAllTests()

   allTrainingReviews = extractReviews(reviews)

   #TEST
   classy = WordWeightClassifier(allTrainingReviews)
   print 'RMSE: {0}'.format(crossValidate(allTrainingReviews, lambda trainSet: WordWeightClassifier(trainSet)))
   sys.exit()

   #print 'RMSE: {0}'.format(crossValidate(allTrainingReviews, lambda trainSet: NBClassifier(trainSet), 4, 0))

   #random.shuffle(allTrainingReviews)
   #classy = NBClassifier(allTrainingReviews)

   trainSet = allTrainingReviews[:-42]
   testSet = allTrainingReviews[-42:]

   classy = NBClassifier(trainSet)

   # TEST
   print 'Accuracy {0}'.format(classy.accuracy(testSet))
   classy.showInfo()
   sys.exit(0)

   for review in tests:
      fileName = review['file']
      filePos = review['position']
      for cat in ['food', 'service', 'venue', 'overall']:
         prediction = classy.classifyDocument(review[cat + 'Review'])
         print '{0}::{1}:{2} -- {3}'.format(fileName, filePos, cat, prediction)
