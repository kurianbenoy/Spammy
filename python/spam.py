# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('spam_ham.csv')
X = dataset.iloc[:, [0]].values
y = dataset.iloc[:, 1].values
# print(dataset.head())
# print(X,y)

# Splitting Dataset into train to test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Converting text to vectors
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(X_train.ravel())
print(vectors)

# print(X_train)

#Label Encoding the dataset
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
print(y_train)

# # Fitting classifier to the Training set
# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(vectors,y_train)

# # Predicting the Test set results
# y_pred = classifier.predict(X_test)

# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)

# CountVectorizer method
# from sklearn.feature_extraction.text import CountVectorizer
# vect = CountVectorizer().fit(X_train.ravel())
# print(len(vect.get_feature_names()[::20]))

