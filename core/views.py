import datetime
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation

from django.shortcuts import render
from django.views.generic import FormView


from .forms import ClassifierForm
from .models import Classifer



def feature_extractor(X,y):
    vectorizer = TfidfVectorizer(max_features=50,ngram_range=(1,2))
    le = preprocessing.LabelEncoder()
    transformed_data = vectorizer.fit_transform(X)
    transformed_label = le.fit_transform(y)
    return vectorizer,transformed_data, transformed_label



def evaluate_models(X,Y):
    '''
    Model selection
    '''
    model = MultinomialNB()
    kfold = cross_validation.KFold(n=len(Y), n_folds=10, random_state=5)
    cv_results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
    print (cv_results.mean(), cv_results.std())
    return model

# Importing the dataset
dataset = pd.read_csv('spam_ham.csv')
X = dataset['text'].values
y = dataset['type'].values

vectorizer , transformed_data, transformed_label = feature_extractor(X,y)
# print(transformed_label)
# model = evaluate_models(transformed_data,transformed_label)

clf = MultinomialNB(alpha=.01)
clf.fit(transformed_data,transformed_label)
test = ["As a valued customer, I am pleased to advise you that following recent review of your Mob No. you are awarded with a £1500 Bonus Prize, call 09066364589"]
vectors_test = vectorizer.transform(test)
pred = clf.predict(vectors_test)
print (pred)



def Home(request):
	form = ClassifierForm()
	if request.method == 'POST':
		form = ClassifierForm(request.POST)
		if form.is_valid():
			timestamp = datetime.datetime.now()
			print(timestamp)
			form = ClassifierForm(request.POST)
			author = form.save(commit=False)
			author.times = timestamp
			author.save()
		else :
			print(form.errors)

	else:
		form = ClassifierForm()

	return render(request,'core/spamsubmission.html',{'form':form })    
