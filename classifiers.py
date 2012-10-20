import math
import sys
import random

import nltk

import features

def crossValidate(documents, classifierCreator = lambda trainSet: NBClassifier(trainSet),  numFolds = 4, seed = None):
   if seed:
      random.seed(seed)

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
         predictions.append(classy.classifyDocument(testDocument[0], testDocument[1]))
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

   def classifyDocument(self, document, realScore = -1):
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

   def classifyDocument(self, document, realScore = -1):
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

class BinaryClassSplitClassifier:
   #def __init__(self, trainingSet, probabilityDist = nltk.probability.ELEProbDist):
   def __init__(self, trainingSet, probabilityDist = nltk.probability.LaplaceProbDist):
      rawClassFeatures = {}

      #TEST
      allFreqs = {}
      classCounts = {}

      for i in range(1, 6):
         rawClassFeatures[i] = set()

      for trainDocument in trainingSet:
         unigrams = set(features.batchStem(features.toUnigrams(trainDocument[0])))

         if not classCounts.has_key(trainDocument[1]):
            classCounts[trainDocument[1]] = 1
         else:
            classCounts[trainDocument[1]] += 1

         for gram in unigrams:
            rawClassFeatures[int(trainDocument[1])].add(gram)

            if not allFreqs.has_key(gram):
               allFreqs[gram] = 1
            else:
               allFreqs[gram] += 1

      self.uniqueFeatures = {}
      for i in range(1, 6):
         uniques = set(rawClassFeatures[i])

         for j in range(1, 6):
            if i != j:
               uniques -= rawClassFeatures[j]

         #print '{0} -- {1}'.format(i, uniques)
         self.uniqueFeatures[i] = uniques

#      for (score, grams) in self.uniqueFeatures.items():
#         print '{0} ({1})'.format(score, classCounts[score])
#         for gram in grams:
#            print '   {0} ({1})'.format(gram, allFreqs[gram])


      #sys.exit()
      #TEST
      #return

      # TODO: try splitting up into multiple docs
      # TRY: all in one doc, with false and true; invert that for false
      # Two docs, only pos

      self.binClassifiers = {}
      for i in range(1, 6):
         allFeatures = [({gram: True}, 'in') for gram in self.uniqueFeatures[i]]

         for j in range(1, 6):
            for gram in self.uniqueFeatures[j]:
               allFeatures.append(({gram: True}, 'out'))

         self.binClassifiers[i] = nltk.NaiveBayesClassifier.train(allFeatures, probabilityDist)


#      self.binClassifiers = {}
#      for i in range(1, 6):
#         inFeatures = {gram: True for gram in self.uniqueFeatures[i]}
#         outFeatures = {}
#
#         for j in range(1, 6):
#            for gram in self.uniqueFeatures[j]:
#               outFeatures[gram] = True
#
#         for outGram in outFeatures.keys():
#            inFeatures[outGram] = False
#
#         for inGram in inFeatures.keys():
#            outFeatures[inGram] = False
#
#         trainSet = [(inFeatures, 'in'), (outFeatures, 'out')]
#         self.binClassifiers[i] = nltk.NaiveBayesClassifier.train(trainSet, probabilityDist)

   def classifyDocument(self, document, realScore = -1):
      order = [1, 2, 3, 5, 4]
      rtn = 4

      grams = set(features.batchStem(features.toUnigrams(document)))

      #TEST
#      intersectionSizes = {}
#      for i in order:
#         if len(self.uniqueFeatures[i]) == 0:
#            intersectionSizes[i] = 0
#         else:
#            #intersectionSizes[i] = len(self.uniqueFeatures[i].intersection(grams)) / float(len(self.uniqueFeatures[i]))
#            intersectionSizes[i] = len(self.uniqueFeatures[i].intersection(grams))
#
#      score = 4
#      maxOccur = 0
#      for (key, occur) in intersectionSizes.items():
#         if occur > maxOccur:
#            score = key
#            maxOccur = occur
#
#      print '{0} -- {1} - {2}'.format(intersectionSizes, score, realScore)
#      return score
#
#      return random.randint(3, 4)

      featureSet = {gram: True for gram in set(features.batchStem(features.toUnigrams(document)))}
      probs = {}
      weights = []
      for i in order:
         prob = self.binClassifiers[i].prob_classify(featureSet)
         probs[i] = prob.prob('in')
         weights.append(prob.prob('in') * i)
         #probs[i] = prob.prob('out')
         #weights.append(prob.prob('out') * i)


#         if self.binClassifiers[i].classify(featureSet) == 'in':
#            rtn = i
#            print 'Found: {0}'.format(i)
#            break

      score = sum(weights) / 5.0
      print '{0} -- {1} - {2}'.format(probs, score, realScore)
      return score


#TODO: Jacardian dist classifier
# Take Jaccardian dist of document to all training ones. Weight by nuimber of docs in both cats.
# Take the class with the highest avg sim (try mean and median)
class SetDistClassifier:
   def __init__(self, trainingSet):
      self.grams = {}

      #TEST
      rawClassFeatures = {}
      for i in range(1, 6):
         rawClassFeatures[i] = set()

      for trainingDocument in trainingSet:
         if not self.grams.has_key(trainingDocument[1]):
            self.grams[trainingDocument[1]] = []
         grams = set(features.batchStem(features.toUnigrams(trainingDocument[0])))
         self.grams[trainingDocument[1]].append(grams)
         rawClassFeatures[trainingDocument[1]] |= grams

      self.uniqueFeatures = {}
      for i in range(1, 6):
         self.uniqueFeatures[i] = set(rawClassFeatures[i])
         for j in range(1, 6):
            if i != j:
               self.uniqueFeatures[i] -= rawClassFeatures[j]

   def classifyDocument(self, document, realScore = -1):
      featureSet = set(features.batchStem(features.toUnigrams(document)))

      scores = {}

      #TEST uniques
      for (classVal, classGrams) in self.grams.items():
         scores[classVal] = []
         for classGram in classGrams:
            scores[classVal].append(nltk.metrics.distance.jaccard_distance(featureSet, classGram))
      #TEST all
      #for i in range(1, 6):
      #   scores[i] = []
      #   scores[i].append(nltk.metrics.distance.jaccard_distance(featureSet, self.uniqueFeatures[i]))

      finalDists = {}
      minDist = 10000
      bestClass = 4
      for (classVal, dists) in scores.items():
         #dist = sum(dists) / float(len(dists))

         ordered = sorted(dists)
         if len(dists) % 2 == 0 and len(dists) > 0:
            dist = (dists[len(dists) / 2] + dists[len(dists) / 2 - 1]) / 2
         else:
            dist = dists[len(dists) / 2]

         finalDists[classVal] = dist
         if dist < minDist:
            minDist = dist
            bestClass = classVal

      print '{0} -- {1} - r{2}'.format(finalDists, bestClass, realScore)
      return bestClass
