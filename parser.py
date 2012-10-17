import os
import re
import sys

TRAINING_DIR = 'training'
TEST_DIR = 'test'

REVIEW_REGEX = '^(?:\<p\>)?(.+)(?:\<\/p\>)?$'
END_REVIEW_REGEX = '^(?:\<p\>)?(.+)(?:\<\/p\>(?:\<\/body\>\<\/html\>)?)?$'

# All occurences of a tag
def allTagRegex(tag):
   return '\<\s*\/?\s*{0}\s*[^\>]*\>'.format(tag)

# Read all of a review file, and return the cleaned contents
def cleanFile(fileObj):
   contents = ''
   line = fileObj.readline()
   while len(line) != 0:
      contents += line.strip() + '\n'
      line = fileObj.readline()

   # span, body, html, strong, and div tags are not informative
   contents = re.sub(allTagRegex('span'), '', contents, flags=re.IGNORECASE)
   contents = re.sub(allTagRegex('body'), '', contents, flags=re.IGNORECASE)
   contents = re.sub(allTagRegex('html'), '', contents, flags=re.IGNORECASE)
   contents = re.sub(allTagRegex('strong'), '', contents, flags=re.IGNORECASE)
   contents = re.sub(allTagRegex('div'), '', contents, flags=re.IGNORECASE)

   # Aldrin and Connor's review seems to be the only ones that cannot be parsed like all the others.
   # If we just remove all the breaks, it should be ok.
   if (re.search('(montana)|(Mee Heng Low is a noodle house with a vague resemblance to Chinese noodles)|(Upper Crust Trattoria is an italian restaurant in San Luis Obispo)|(Moe\'s Smokehouse BBQ is a BBQ restaurant that primarily serves burgers)|(Connor Lange)|(Firestone Grill is usually extremely busy during the evening)|(One of the most important aspects of F\.McLintocks is the atmosphere of the venue)', contents, flags=re.IGNORECASE)):
      contents = re.sub(allTagRegex('br'), ' ', contents, flags=re.IGNORECASE)

   contents = re.sub(allTagRegex('p'), '\n', contents, flags=re.IGNORECASE)
   contents = re.sub(allTagRegex('br'), '\n', contents, flags=re.IGNORECASE)

   # Replace useless random characters.
   # c2 a0 = nbsp
   # e2 80 9c = left double quote
   # e2 80 9d = right double quote
   # e2 80 99 = right single quote
   # c3 a9 = e with accent aigu
   # e2 80 93 = EN dash
   contents = re.sub('\xc2\xa0', '', contents)
   contents = re.sub('\xe2\x80\x9c', '"', contents)
   contents = re.sub('\xe2\x80\x9d', '"', contents)
   contents = re.sub('\xe2\x80\x99', '\'', contents)
   contents = re.sub('\xc3\xa9', 'e', contents)
   contents = re.sub('\xe2\x80\x93', '', contents)

   cleanContents = ''
   # Make another pass and verify that all the parts are there
   for line in contents.split('\n'):
      line = line.strip()
      if len(line) != 0:
         # If this is the not the first line, and it is a reviewers name. Then put in another \n.
         #if len(cleanContents) != 0 and line.startswith('REVIEWER'):
         #   cleanContents += '\n'

         cleanContents += line + '\n'

   return re.sub('\n$', '', cleanContents)

def consValueRegex(name):
   return '{0}:\s*([^\<]+)(\<)?'.format(name)

def grabConsValue(name, fileObj):
   line = fileObj.readline().strip()
   match = re.search(consValueRegex(name), line)
   if not match:
      sys.stderr.write('Error on the {0} line in {1}: {2}\n'.format(name, fileObj.name, line))
      sys.exit(1)
   return match.group(1)

def grabPara(lastPara, fileObj):
   regex = REVIEW_REGEX
   if lastPara:
      regex = END_REVIEW_REGEX

   line = fileObj.readline().strip()
   match = re.search(regex, line)
   if not match:
      sys.stderr.write('Error in {0}: {1}\n'.format(fileObj.name, line))
      sys.exit(1)
   return match.group(1)

def getConsValue(objectKey, textValue, line, review):
   match = re.search(consValueRegex(textValue), line)
   if not match:
      raise
   review[objectKey] = match.group(1)

# Always returns an array if revirews.
# Expects cleaned input
def stringToReviews(content, baseFile):
   rtn = []

   lines = content.split('\n')
   for i in range(0, int(len(lines) / 13)):
      try:
         review = {}

         getConsValue('reviewer', 'REVIEWER', lines[i * 13 + 0], review)
         getConsValue('restaurant', 'NAME', lines[i * 13 + 1], review)

         # Consume address and city
         # lines[i * 13 + 2]
         # lines[i * 13 + 3]

         getConsValue('foodScore', 'FOOD', lines[i * 13 + 4], review)
         getConsValue('serviceScore', 'SERVICE', lines[i * 13 + 5], review)
         getConsValue('venueScore', 'VENUE', lines[i * 13 + 6], review)
         getConsValue('overallScore', 'RATING', lines[i * 13 + 7], review)

         # Consume the "WRITTEN REVIEW" line
         # lines[i * 13 + 8]

         # Paragraphs have already been properly formatted and stripped
         review['foodReview'] = lines[i * 13 + 9]
         review['serviceReview'] = lines[i * 13 + 10]
         review['venueReview'] = lines[i * 13 + 11]
         review['overallReview'] = lines[i * 13 + 12]

         review['file'] = baseFile
         review['position'] = i

         rtn.append(review)
      except:
         pass

   return rtn

def readReviews(targetDir):
   reviews = []
   listing = os.listdir(targetDir)
   for baseFile in listing:
      # Skip hidden, stupid mac files.
      if baseFile.startswith('.'):
         continue

      fileName = targetDir + '/' + baseFile
      fileObj = open(fileName, 'r')

      content = cleanFile(fileObj)
      reviews += stringToReviews(content, baseFile)

   return reviews

def readAllReviews():
   return readReviews(TRAINING_DIR)

def readAllTests():
   return readReviews(TEST_DIR)

if __name__ == '__main__':
   #fileObj = open('training/Matthew Parker_53951.html', 'r')
   #fileObj = open('raw/training/Aldrin Montana_12349.html', 'r')
   #fileObj = open('raw/training/Jacob Muir_266510.html', 'r')
   #fileObj = open(sys.argv[1], 'r')
   #cleanFile(fileObj)
   reviews = readAllReviews()
   print reviews
