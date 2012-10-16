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

"""
for minOccur in range(1, 10):
   for maxOccurPercent in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
      classy = NBClassifier(trainSet, features.FeatureSetGenerator(minOccur, maxOccurPercent))
      print 'MinOccur: {0}, MaxOccurPercent: {1}, Accuracy: {2}'.format(minOccur, maxOccurPercent, classy.accuracy(testSet))
"""

#classy = NBClassifier(trainSet)
#print classy.accuracy(testSet)
#classy.classifier().show_most_informative_features(30)

"""
allReviews = []
for review in reviews:
   for cat in ['food', 'service', 'venue', 'overall']:
      allReviews.append(review[cat + 'Review'])
"""

""" Binary classifiers below here
# [ ( review para, score ) ]
allPairs = []
for review in reviews:
   for cat in ['food', 'service', 'venue', 'overall']:
      allPairs.append((review[cat + 'Review'], int(review[cat + 'Score'])))

#random.shuffle(allPairs)
trainingReviews, testingReviews = allPairs[0:-42], allPairs[-42:]

fullTests = {}
fullTrainings = {}
for i in range(1, 6):
   fullTrainings[i] = []
   fullTests[i] = []

   for reviewPair in trainingReviews:
      score = int(reviewPair[1])
      if score == i:
         fullTrainings[i].append((features.toFeatures(reviewPair[0]), 'in'))
      else:
         fullTrainings[i].append((features.toFeatures(reviewPair[0]), 'not'))

   for testPair in testingReviews:
      score = int(testPair[1])
      if score == i:
         fullTests[i].append((features.toFeatures(testPair[0]), 'in'))
      else:
         fullTests[i].append((features.toFeatures(testPair[0]), 'not'))


# {binary class => trained classifier}
classifiers = {}
for i in range(1, 6):
   classifiers[i] = nltk.NaiveBayesClassifier.train(fullTrainings[i])

right = 0
wrong = 0
for index in range(0, len(testingReviews)):
   realScore = testingReviews[index][1]

   maxProb = -1.0
   bestClass = 3

   for i in range(1, 6):
      if classifiers[i].classify(fullTests[i][index][0]) == 'in':
         bestClass = i
         break


#   for i in range(1, 6):
#      print classifiers[i].classify(fullTests[i][index][0])
#
#      prob = classifiers[i].prob_classify(fullTests[i][index][0]).prob('in')
#      print '{0} --- {1}'.format(i, prob)
#
#      if (prob > maxProb):
#         print 'trans'
#         maxProb = prob
#         bestClass = i

   print 'RealScore: {0}, Guess: {1}'.format(realScore, bestClass)

   if realScore == bestClass:
      right += 1
   else:
      wrong += 1

print 'Right: {0}, Wrong: {1}, Accuracy: {2}'.format(right, wrong, right / (right + wrong))
"""

"""
res = {}
for algorithm in ['GIS']:
#for algorithm in ['GIS', 'IIS']:
#for algorithm in ['GIS', 'IIS', 'CG', 'BFGS', 'Powell', 'LBFGSB', 'Nelder-Mead']:
   classifier = nltk.MaxentClassifier.train(trainSet, algorithm, 0)
   acc = nltk.classify.accuracy(classifier, testSet)
   res[algorithm] = acc

print res
"""
