# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation


def feature_extractor(X,y):
    vectorizer = TfidfVectorizer(max_features=50,ngram_range=(1,2))
    le = preprocessing.LabelEncoder()
    transformed_data = vectorizer.fit_transform(X)
    transformed_label = le.fit_transform(y)
    return transformed_data, transformed_label



def evaluate_models(X,Y):
    '''
    Model selection
    '''
    model = MultinomialNB()
    kfold = cross_validation.KFold(n=len(Y), n_folds=10, random_state=5)
    cv_results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
    print (cv_results.mean(), cv_results.std())


# Importing the dataset
dataset = pd.read_csv('spam_ham.csv')
X = dataset['text'].values
y = dataset['type'].values

transformed_data , transformed_label = feature_extractor(X,y)
# print(transformed_data[0])
evaluate_models(transformed_data,transformed_label)

