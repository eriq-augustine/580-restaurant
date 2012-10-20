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

if __name__ == '__main__':
   reviews = parser.readAllReviews()
   tests = parser.readAllTests()

   allTrainingReviews = extractReviews(reviews)

   #TEST
   #print 'RMSE: {0}'.format(classifiers.crossValidate(allTrainingReviews, lambda trainSet: classifiers.SetDistClassifier(trainSet)))
   #sys.exit()

   #lassy = classifiers.WordWeightClassifier(allTrainingReviews)
   #classy = classifiers.BinaryClassSplitClassifier(allTrainingReviews)
   #print 'RMSE: {0}'.format(classifiers.crossValidate(allTrainingReviews, lambda trainSet: classifiers.WordWeightClassifier(trainSet)))
   #print 'RMSE: {0}'.format(classifiers.crossValidate(allTrainingReviews, lambda trainSet: classifiers.BinaryClassSplitClassifier(trainSet)))
   #sys.exit()

   #print 'RMSE: {0}'.format(classifiers.crossValidate(allTrainingReviews, lambda trainSet: classifiers.NBClassifier(trainSet), 4, 0))

   #random.shuffle(allTrainingReviews)
   #classy = classifiers.NBClassifier(allTrainingReviews)

   trainSet = allTrainingReviews[:-42]
   testSet = allTrainingReviews[-42:]

   classy = classifiers.NBClassifier(trainSet)

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
