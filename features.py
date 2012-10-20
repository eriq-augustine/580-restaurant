import nltk
import re
import sys

class NoStemmer:
   def stem(self, word):
      return word

# This is the core function for stripping out punct and splitting into words
def toUnigrams(document):
   document = document.lower()
   document = re.sub('\[|\]|\(|\)|,|\.|:|;|"|~|\/|\\|(--)|#', ' ', document)
   document = re.sub('!', ' __exclamation_point__ ', document)
   document = re.sub('\?', ' __quertion__mark__ ', document)
   document = re.sub("'|-|\$", '', document)

   return nltk.tokenize.word_tokenize(document)

def batchStem(unigrams, stemmer = nltk.stem.PorterStemmer()):
   return [stemmer.stem(gram) for gram in unigrams]

# return hash with things that we are looking for,look at FSG::defineAllFeatures
def posGroup(document):
   conj = 0
   nouns = 0
   adv = 0
   adj = 0
   w = 0
   v = 0
   uh = 0

   for (word, tag) in nltk.pos_tag(document):
      if tag == 'IN':
         conj += 1
      elif tag.startswith('N'):
         nouns += 1
      elif tag.startswith('J'):
         adj += 1
      elif tag.startswith('RB'):
         adv += 1
      elif tag.startswith('W'):
         w += 1
      elif tag.startswith('V'):
         v += 1
      elif tag.startswith('U'):
         uh += 1

   features = {'<$Num Conjunctions$>': conj,
               '<$Num Nouns$>': nouns,
               '<$Num Adverbs$>': adv,
               '<$Num Adjectives$>': adj,
               '<$Num Verbs$>': adj,
               '<$Num Interjections$>': adj,
               '<$Num W-guys$>': w}
   print document
   print features
   sys.exit()
   return features

class FeatureSetGenerator:
   #def __init__(self, minOccur = 4, maxOccurPercent = 0.50, stemmer = nltk.stem.PorterStemmer()):
   def __init__(self, minOccur = 2, maxOccurPercent = 0.80, stemmer = nltk.stem.PorterStemmer()):
      # features must be in 5 docs before being included
      self.MIN_OCCURENCE = minOccur
      # features occuring in >20% of docs are stop words
      self.MAX_OCCURENCE_PERCENT = maxOccurPercent
      self.stemmer = stemmer
      self.definedFeatures = set()

   # This takes all the training data, and specifies which features are allowd.
   # We have to be very careful with our small dataset not to let some outliers pull us around.
   def defineAllFeatures(self, documents):
      self.definedFeatures = set()

      # These are in every document, but are numeric, so let them in.
      self.definedFeatures.add('<$Unique Words$>')
      self.definedFeatures.add('<$Average Word Length$>')
      self.definedFeatures.add('<$Num Words$>')
      self.definedFeatures.add('<$Text Length$>')
      self.definedFeatures.add('<$Num Conjunctions$>')
      self.definedFeatures.add('<$Num Nouns$>')
      self.definedFeatures.add('<$Num Adverbs$>')
      self.definedFeatures.add('<$Num Adjectives$>')
      self.definedFeatures.add('<$Num W-guys$>')
      self.definedFeatures.add('<$Num Interjections$>')
      self.definedFeatures.add('<$Num Verbs$>')

      # { feature => count }
      counts = {}

      for document in documents:
         features = self.toFullFeatures(document)
         for feature in features.keys():
            counts[feature] = counts.get(feature, 0) + 1

      numDocs = len(documents)

      for (feature, count) in counts.items():
         if count >= self.MIN_OCCURENCE and float(count) / numDocs < self.MAX_OCCURENCE_PERCENT:
            self.definedFeatures.add(feature)

   # Set compression sould NOT be used for Naive Bayes!
   def toFullFeatures(self, document, observeDefinedFeatures = False, compressSet = False):
      features = {}

      unigrams = toUnigrams(document)
      stemmedUnigrams = batchStem(unigrams, self.stemmer)

      avgLen = 0
      for index in range(0, len(unigrams)):
         if not observeDefinedFeatures or stemmedUnigrams[index] in self.definedFeatures:
            features[stemmedUnigrams[index]] = True
         avgLen += len(unigrams[index])
      avgLen /= float(len(unigrams))

      #bigrams = nltk.util.bigrams(stemmedUnigrams)
      #for gram in bigrams:
      #   if not observeDefinedFeatures or gram in self.definedFeatures:
      #      features[gram] = True

      unique = len(set(unigrams))

      # NB doesn't deal well with numeric attributes, so rangeify them.
      #features['<$Unique Words$>'] = self.rangeify(unique, 10)
      #features['<$Average Word Length$>'] = self.rangeify(avgLen, 2)
      #features['<$Text Length$>'] = self.rangeify(len(document), 50)
      #features['<$Num Words$>'] = self.rangeify(len(unigrams), 20)

      if observeDefinedFeatures:
         features['<$Unique Words$>'] = unique
         features['<$Average Word Length$>'] = avgLen
         features['<$Text Length$>'] = len(document)
         features['<$Num Wdefined>'] = len(unigrams)
         features.update(posGroup(document))

      if (not compressSet) and observeDefinedFeatures:
         for definedFeature in self.definedFeatures:
            if not features.has_key(definedFeature):
               features[definedFeature] = False

      return features

   def rangeify(self, val, step):
      return '{0}-{1}'.format(val / step * step, (val + step) / step * step)

   # Break into full features, but then only include the defined features.
   def toFeatures(self, document):
      return self.toFullFeatures(document, True)
