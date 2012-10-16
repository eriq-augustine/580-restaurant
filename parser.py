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

   # span, body, html tags are not informative
   contents = re.sub(allTagRegex('span'), '', contents, flags=re.IGNORECASE)
   contents = re.sub(allTagRegex('body'), '', contents, flags=re.IGNORECASE)
   contents = re.sub(allTagRegex('html'), '', contents, flags=re.IGNORECASE)

   # Before any other nonsense, replace any breaks INSIDE <p> with spaces. Thanks, Aldrin
   #\<\s*p\s*[^\>]*\>(.*)\<\s*\/\s*p\s*[^\>]*\>

   contents = re.sub(allTagRegex('p'), '\n', contents, flags=re.IGNORECASE)
   contents = re.sub(allTagRegex('br'), '\n', contents, flags=re.IGNORECASE)

   cleanContents = ''
   # Make another pass and verify that all the parts are there
   #TODO
   for line in contents.split('\n'):
      line = line.strip()
      if len(line) != 0:
         cleanContents += line + '\n'

   print cleanContents
   sys.exit(0)

   return cleanContents

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

def readReview(fileObj):
   if fileObj.closed:
      return False

   review = {}
   review['reviewer'] = grabConsValue('REVIEWER', fileObj)
   review['restaurant'] = grabConsValue('NAME', fileObj)

   # Consume address and city
   fileObj.readline()
   fileObj.readline()

   review['foodScore'] = grabConsValue('FOOD', fileObj)
   review['serviceScore'] = grabConsValue('SERVICE', fileObj)
   review['venueScore'] = grabConsValue('VENUE', fileObj)
   review['overallScore'] = grabConsValue('RATING', fileObj)

   # Consume the "WRITTEN REVIEW" line
   fileObj.readline()

   review['foodReview'] = grabPara(False, fileObj)
   review['serviceReview'] = grabPara(False, fileObj)
   review['venueReview'] = grabPara(False, fileObj)
   review['overallReview'] = grabPara(False, fileObj)

#   # Consume a final line, this will do nothing if it is one review per file,
#   #  but if there are multiple reviews in a file, it will take the sep space.
   line = fileObj.readline()
   if len(line) == 0:
      fileObj.close()

   fileObj.close()

   return review

def readReviews(targetDir):
   reviews = []
   listing = os.listdir(targetDir)
   for baseFile in listing:
      fileName = targetDir + '/' + baseFile
      fileObj = open(fileName, 'r')

      count = 0

      while True:
         review = readReview(fileObj)
         if not review:
            break
         review['file'] = baseFile
         review['position'] = count
         reviews.append(review)

         count += 1
   return reviews

def readAllReviews():
   return readReviews(TRAINING_DIR)

def readAllTests():
   return readReviews(TEST_DIR)

if __name__ == '__main__':
   #fileObj = open('training/Matthew Parker_53951.html', 'r')
   fileObj = open('raw/training/Aldrin Montana_12349.html', 'r')
   cleanFile(fileObj)
   #reviews = readAllReviews()
   #print reviews
