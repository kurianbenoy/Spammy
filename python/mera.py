# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection
from sklearn import cross_validation


# Importing the dataset
dataset = pd.read_csv('spam_ham.csv')
X = dataset['text'].values
y = dataset['type'].values

# Feauture Extraction  

vectorizer = TfidfVectorizer(max_features=50,ngram_range=(1,2))
le = preprocessing.LabelEncoder()
transformed_data = vectorizer.fit_transform(X)
transformed_label = le.fit_transform(y)


model = MultinomialNB()
model.fit(transformed_data,transformed_label)

print(model.predict(vectorizer.transform(['spammer nanugghty idiot','not a problem']))  )

# kfold = cross_validation.KFold(n=len(y), n_folds=10, random_state=5)
# cv_results = cross_validation.cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
# print (cv_results.mean(), cv_results.std())