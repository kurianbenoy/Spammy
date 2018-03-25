# Program written to test csv file format

from textblob.classifiers import NaiveBayesClassifier
import csv
import sys

simpletext = []
sometext = []

with open('spam_ham.csv','r') as simple:
    sometext = csv.reader(simple)
    for row in sometext:
    # print 'row1 ', row
        tblb = ()
        tblb = (row[0],row[1])
        simpletext.append(tblb)

    nbcl = NaiveBayesClassifier(simpletext)

testworthy = []
testable = []
with open('spam_ham.csv','r') as testthis:
    testable = csv.reader(testthis)
    for row in testable:
    # print 'testable ', row
        tblb = ()
        tblb = (row[0],row[1])
        testworthy.append(tblb)

for row in testworthy:
    print (row[0], ' classified as ', nbcl.classify(row[0]), ' with ground_truth ', row[1])