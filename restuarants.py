import math
import sys
import random

import nltk

import parser
import features

def crossValidate(documents, numFolds = 4):
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

      classy = NBClassifier(trainSet)
      predictions = classy.classifier().batch_classify(classy.unlabeledDocsToFeatures(testSet))
      rmse += calcRmse(predictions, [doc[1] for doc in testSet])

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

def extractReviews(reviews):
   rtn = []
   for review in reviews:
      for cat in ['food', 'service', 'venue', 'overall']:
         rtn.append((review[cat + 'Review'], int(review[cat + 'Score'])))
   return rtn

reviews = parser.readAllReviews()
tests = parser.readAllTests()

allTrainingReviews = extractReviews(reviews)

print 'RMSE: {0}'.format(crossValidate(allTrainingReviews))

random.shuffle(allTrainingReviews)
classy = NBClassifier(allTrainingReviews)

for review in tests:
   fileName = review['file']
   filePos = review['position']
   for cat in ['food', 'service', 'venue', 'overall']:
      prediction = classy.classifyDocument(review[cat + 'Review'])
      print '{0}::{1}:{2} -- {3}'.format(fileName, filePos, cat, prediction)
