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
   contents = re.sub(allTagRegex('strong'), '', contents, flags=re.IGNORECASE)
   contents = re.sub(allTagRegex('div'), '', contents, flags=re.IGNORECASE)

   # Aldrin's review seems to be the only ones that cannot be parsed like all the others.
   # If we just remove all the breaks, it should be ok.
   #if (re.search(('(aldrin)|(montana)|' +
   #               '(Mee Heng Low is a noodle house with a vague resemblance to Chinese noodles)|' +
   #               '(Upper Crust Trattoria is an italian restaurant in San Luis Obispo)|'+
   #               '(Moe\'s Smokehouse BBQ is a BBQ restaurant that primarily serves burgers)',
   #               contents, flags=re.IGNORECASE))):
   if (re.search('(montana)|(Mee Heng Low is a noodle house with a vague resemblance to Chinese noodles)|(Upper Crust Trattoria is an italian restaurant in San Luis Obispo)|(Moe\'s Smokehouse BBQ is a BBQ restaurant that primarily serves burgers)', contents, flags=re.IGNORECASE)):
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
         if len(cleanContents) != 0 and line.startswith('REVIEWER'):
            cleanContents += '\n'

         cleanContents += line + '\n'

   # TEST
   sys.stdout.write(cleanContents)

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

   # Consume a final line, this will do nothing if it is one review per file,
   #  but if there are multiple reviews in a file, it will take the sep space.
   line = fileObj.readline()
   if len(line) == 0:
      fileObj.close()

   fileObj.close()

   return review

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
      continue

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
   #fileObj = open('raw/training/Aldrin Montana_12349.html', 'r')
   #fileObj = open('raw/training/Jacob Muir_266510.html', 'r')
   #fileObj = open(sys.argv[1], 'r')
   #cleanFile(fileObj)
   reviews = readAllReviews()
   #print reviews
