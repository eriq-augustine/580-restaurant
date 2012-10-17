import math
import sys
import random

import nltk

import parser
import features

def crossValidate(documents, classifierCreator = lambda trainSet: NBClassifier(trainSet),  numFolds = 4):
   ordering = [i for i in range(0, len(documents))]
   random.shuffle(ordering)

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
      self.classy = nltk.NaiveBayesClassifier.train(self.labeledDocsToFeatures(trainingSet))

   def classifyDocument(self, document):
      return self.classy.classify(self.fsg.toFeatures(document))

   def classifyFeatures(self, features):
      return self.classy.classify(features)

   def classifier(self):
      return self.classy

   def accuracy(self, testSet):
      return nltk.classify.accuracy(self.classy, self.labeledDocsToFeatures(testSet))

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
         if score < mean:
            #negatives.append(self.meanScore - score)
            #negatives.append(mean - score)
            negatives.append(score)
         else:
            #positives.append(score - self.meanScore)
            #positives.append(score - mean)
            positives.append(score)

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
      positiveScore = sum(positives) * len(scores) / len(positives)
      negativeScore = sum(negatives) * len(scores) / len(negatives)
      evenScore = positiveScore / negativeScore * self.meanScore
      print 'Even -- Positive: {0}, Negative: {1}, Even Score: {2}'.format(positiveScore, negativeScore, evenScore)

      sortedScores = sorted(scores)
      median = sortedScores[int(len(sortedScores) / 2)]
      print 'Median: {0}'.format(median)

      #print 'Mean: {0}, CrazyScore: {1}'.format(mean, crazyScore)
      print 'Mean: {0}, CrazyScore: {1}, Real Score: {2}'.format(mean, crazyScore, document[1])

      return median

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

   print 'RMSE: {0}'.format(crossValidate(allTrainingReviews, lambda trainSet: NBClassifier(trainSet)))

   random.shuffle(allTrainingReviews)
   classy = NBClassifier(allTrainingReviews)

   for review in tests:
      fileName = review['file']
      filePos = review['position']
      for cat in ['food', 'service', 'venue', 'overall']:
         prediction = classy.classifyDocument(review[cat + 'Review'])
         print '{0}::{1}:{2} -- {3}'.format(fileName, filePos, cat, prediction)
