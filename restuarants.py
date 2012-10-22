import math
import sys
import random

import nltk

import parser
import features
import classifiers

def extractReviews(reviews):
   rtn = []
   for review in reviews:
      for cat in ['food', 'service', 'venue', 'overall']:
         rtn.append((review[cat + 'Review'], int(review[cat + 'Score'])))
   return rtn

def extractNames(reviews):
   rtn = []
   for review in reviews:
      for cat in ['food', 'service', 'venue', 'overall']:
         rtn.append('{0}::{1}:{2}'.format(review['file'], review['position'], cat))
   return rtn

def ex1(reviews, tests, predictionInfo):
   allTrainingReviews = extractReviews(reviews)
   allTrainingNames = extractNames(reviews)

   rmse = classifiers.crossValidate(allTrainingReviews, lambda trainSet: classifiers.TwoClassifier(trainSet), 4, allTrainingNames)
   print 'Average RMS error rate on validation set: {0}\n'.format(rmse)

   classy = classifiers.TwoClassifier(allTrainingReviews)
   for review in tests:
      key = '{0}::{1}'.format(review['file'], review['position'])
      for cat in ['food', 'service', 'venue', 'overall']:
         prediction = classy.classifyDocument(review[cat + 'Review'])
         predictionInfo[key][cat] = prediction

def ex2(reviews, tests, predictionInfo):
   # Transforms the documents some
   transReviews = []
   names = []
   for review in reviews:
      doc = {}
      for cat in ['food', 'service', 'venue', 'overall']:
         doc[cat] = int(review[cat + 'Score'])
      transReviews.append((doc, int(review['overallScore'])))
      names.append('{0}::{1}'.format(review['file'], review['position']))

   rmse = classifiers.crossValidate(transReviews, lambda trainSet: classifiers.OverallClassifier(trainSet), 4, names)
   print 'Average RMS error rate on validation set: {0}\n'.format(rmse)

   classy = classifiers.OverallClassifier(transReviews)
   for review in tests:
      key = '{0}::{1}'.format(review['file'], review['position'])
      # Get the info from the predictions from ex1
      doc = {}
      for cat in ['food', 'service', 'venue']:
         doc[cat] = predictionInfo[key][cat]
      prediction = classy.classifyDocument(doc)
      predictionInfo[key]['loneOverall'] = prediction
  
def ex3(reviews, tests, predictionInfo):
   # Transforms the documents some
   transReviews = []
   names = []
   for review in reviews:
      for cat in ['food', 'service', 'venue', 'overall']:
         transReviews.append((review[cat + 'Review'], review['reviewer']))
         names.append('{0}::{1}'.format(review['file'], review['position']))
   '''
   for review in reviews:
      doc = ''
      for cat in ['food', 'service', 'venue', 'overall']:
         doc += ' ' + review[cat + 'Review']
      transReviews.append((doc, review['reviewer']))
      names.append('{0}::{1}'.format(review['file'], review['position']))
   '''

   rmse = classifiers.crossValidate(transReviews, lambda trainSet: classifiers.NBClassifier(trainSet), 4, names, True)
   print 'Average RMS error rate on validation set: {0}\n'.format(rmse)

   classy = classifiers.NBClassifier(transReviews)
   for test in tests:
      key = '{0}::{1}'.format(test['file'], test['position'])
      predictions = {}
      for cat in ['food', 'service', 'venue', 'overall']:
         prediction = classy.classifyDocument(review[cat + 'Review'])
         if not predictions.has_key(prediction):
            predictions[prediction] = 1
         else:
            predictions[prediction] += 1
      if len(predictions) == 4:
         # Random plz
         finalPrediction = predictions.keys()[random.randint(0, len(predictions) - 1)]
      else:
         finalPrediction = max(predictions, key=predictions.get)
      predictionInfo[key]['reviewer'] = finalPrediction

def doAssignment(reviews, tests):
   predictionInfo = {}

   for test in tests:
      predictionInfo['{0}::{1}'.format(test['file'], test['position'])] = {}

   ex1(reviews, tests, predictionInfo)
   ex2(reviews, tests, predictionInfo)
   ex3(reviews, tests, predictionInfo)

   print predictionInfo

if __name__ == '__main__':
   reviews = parser.readAllReviews()
   tests = parser.readAllTests()
   print '{0} Labeled Reviews and {1} Unlabeled Reviews Found.'.format(len(reviews), len(tests))

   doAssignment(reviews, tests)
   sys.exit()

   allTrainingReviews = extractReviews(reviews)

#   res = {}
#   bestRes = (10, '')
#   for i in range(1, 10):
#      for j in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#         res['{0}-{1}'.format(i, j)] = classifiers.crossValidate(allTrainingReviews, lambda trainSet: classifiers.WordWeightClassifier(trainSet, i, j))
#         if res['{0}-{1}'.format(i, j)] < bestRes[0]:
#            bestRes = (res['{0}-{1}'.format(i, j)], '{0}-{1}'.format(i, j))
#
#   print res
#   print bestRes
#   sys.exit()

   print 'RMSE: {0}'.format(classifiers.crossValidate(allTrainingReviews, lambda trainSet: classifiers.TwoClassifier(trainSet)))
   sys.exit()

   print 'RMSE: {0}'.format(classifiers.crossValidate(allTrainingReviews, lambda trainSet: classifiers.WordWeightClassifier(trainSet)))
   sys.exit()

   print 'RMSE: {0}'.format(classifiers.crossValidate(allTrainingReviews, lambda trainSet: classifiers.SentiClassifier(trainSet)))
   sys.exit()

   #TEST
   setDistRmse = classifiers.crossValidate(allTrainingReviews, lambda trainSet: classifiers.SetDistClassifier(trainSet))
   wwRmse = classifiers.crossValidate(allTrainingReviews, lambda trainSet: classifiers.WordWeightClassifier(trainSet))
   binaryRmse = classifiers.crossValidate(allTrainingReviews, lambda trainSet: classifiers.BinaryClassSplitClassifier(trainSet))
   nbRmse = classifiers.crossValidate(allTrainingReviews, lambda trainSet: classifiers.NBClassifier(trainSet), 4, 0)

   print 'SetDist: {0}, WW: {1}, Binary: {2}, NB: {3}'.format(setDistRmse, wwRmse, binaryRmse, nbRmse)

   sys.exit()

   #classy = classifiers.WordWeightClassifier(allTrainingReviews)
   #classy = classifiers.BinaryClassSplitClassifier(allTrainingReviews)
   #print 'RMSE: {0}'.format(classifiers.crossValidate(allTrainingReviews, lambda trainSet: classifiers.WordWeightClassifier(trainSet)))
   #print 'RMSE: {0}'.format(classifiers.crossValidate(allTrainingReviews, lambda trainSet: classifiers.BinaryClassSplitClassifier(trainSet)))
   #sys.exit()

   #print 'RMSE: {0}'.format(classifiers.crossValidate(allTrainingReviews, lambda trainSet: classifiers.NBClassifier(trainSet), 4, 0))

   #random.shuffle(allTrainingReviews)
   #classy = classifiers.NBClassifier(allTrainingReviews)

   trainSet = allTrainingReviews[:-42]
   testSet = allTrainingReviews[-42:]

   #classy = classifiers.NBClassifier(trainSet)
   classy = classifiers.MaxEntClassifier(trainSet)

   # TEST
   print 'Accuracy {0}'.format(classy.accuracy(testSet))
   classy.showInfo()
   #print 'RMSE: {0}'.format(classifiers.crossValidate(allTrainingReviews, lambda trainSet: classifiers.NBClassifier(trainSet)))
   print 'RMSE: {0}'.format(classifiers.crossValidate(allTrainingReviews, lambda trainSet: classifiers.MaxEntClassifier(trainSet)))
   sys.exit(0)

   for review in tests:
      fileName = review['file']
      filePos = review['position']
      for cat in ['food', 'service', 'venue', 'overall']:
         prediction = classy.classifyDocument(review[cat + 'Review'])
         print '{0}::{1}:{2} -- {3}'.format(fileName, filePos, cat, prediction)
