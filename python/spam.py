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

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)

# CountVectorizer method
# from sklearn.feature_extraction.text import CountVectorizer
# vect = CountVectorizer().fit(X_train.ravel())
# print(len(vect.get_feature_names()[::20]))

# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(random_state = 0)
# classifier.fit(X_train,y_train)