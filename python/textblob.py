import numpy as np
# from textblob import TextBlob
# from textblob.classifiers import NaiveBayesClassifier
from textblob.taggers import NLTKTagger
# Training Data

print ("Hello")
train = [
     ('I love this sandwich.', 'Ham'),
     ('this is an amazing place!', 'Ham'),
     ('I feel very good about these beers.', 'Ham'),
     ('this is my best work.', 'Ham'),
     ("what an awesome view", 'Ham'),
     ('I do not like this restaurant', 'Spam'),
     ('I am tired of this stuff.', 'Spam'),
     ("I can't deal with this", 'Spam'),
     ('he is my sworn enemy!', 'Spam'),
     ('my boss is horrible.', 'Spam')
 ]


# actual training of data
cl = NaiveBayesClassifier(train)

text = raw_input("Enter the text to classify")
res = cl.classify(text)
print(res)
res_percentage = cl.classify(text)
print("\n")
print(round(res_percentage("ham"),2) + "\n" + round(res_percentage("spam"),2))


