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


# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# CountVectorizer method
# from sklearn.feature_extraction.text import CountVectorizer
# vect = CountVectorizer().fit(X_train.ravel())
# print(len(vect.get_feature_names()[::20]))

# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(random_state = 0)
# classifier.fit(X_train,y_train)