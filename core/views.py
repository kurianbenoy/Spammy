# Importing the libraries

import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing,model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import cross_validation
from nltk.corpus import stopwords

# Django imports
from django.shortcuts import render
from django.views.generic import FormView



try:
	from .forms import ClassifierForm
	from .models import Classifer
except:
	from forms import ClassifierForm
	from models import Classifer
	

"""
The following are function used for predicting whether the given message is Spam or Ham

"""
def feature_extractor(X,y):
    """
        Using bag of words model using TfidfVectorizer and LabelEncoder to convert the dataset
    """
    mystop_words = set(stopwords.words('english'))
    # print(mystop_words)
    vectorizer = TfidfVectorizer(max_features=50,ngram_range=(1,2),stop_words=mystop_words)
    le = preprocessing.LabelEncoder()
    transformed_data = vectorizer.fit_transform(X)
    transformed_label = le.fit_transform(y)
    return vectorizer,transformed_data, transformed_label

MLPC=MLPClassifier(activation='logistic',solver='lbfgs',random_state=1)
ABC=AdaBoostClassifier(n_estimators=50,learning_rate=1,algorithm='SAMME.R',random_state=None)
RFC=RandomForestClassifier(n_estimators=10,criterion='gini',n_jobs=-1)




def Classifier(X,y):
	X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2)
#classif

	classifier=None
	score=0.95
	for clf in [MLPC,ABC,RFC]:
		model=clf.fit(X_train,y_train)
		if model.score(X_test,y_test)>score:
			score=model.score(X_test,y_test)
			print(score)
			print(clf)		
			classifier=clf
	if classifier==None:
		classifier=svm.SVC(gamma=0.1,C=1)
	

	classifier.fit(X_train,y_train)

	return classifier


def evaluate_models(X,Y):
    '''
    Model selection
    '''
    X=preprocessing.LabelEncoder().fit_transform(X)
    X=X.reshape(-1,1)
    model=Classifier(X,y)
    kfold = cross_validation.KFold(n=len(Y), n_folds=10, random_state=5)
    cv_results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
    print (cv_results.mean(), cv_results.std())
    return model

# Importing the dataset
dataset = pd.read_csv('/home/srinidhi/Spammy/Dataset/spam_ham.csv')
X = dataset['text'].values
y = dataset['type'].values

vectorizer , transformed_data, transformed_label = feature_extractor(X,y)

X=preprocessing.LabelEncoder().fit_transform(X)
X=X.reshape(-1,1)
clf =Classifier(X,y)
clf.fit(transformed_data,transformed_label)
test = ["As a valued customer, I am pleased to advise you that following recent review of your Mob No. you are awarded with a £1500 Bonus Prize, call 09066364589"]
vectors_test = vectorizer.transform(test)



def Home(request):
    form = ClassifierForm()
    if request.method == 'POST':
        form = ClassifierForm(request.POST)
        if form.is_valid():
            timestamp = datetime.datetime.now()
            # print(timestamp)
            predict = form.cleaned_data['inputtext']
            predicton_text = [predict]
            vectors_predtest = vectorizer.transform(predicton_text)
            forcast = clf.predict(vectors_predtest)
            # result = list(map(int,forcast))
            # print(result)
            print(forcast)
            author = form.save(commit=False)
            author.times = timestamp
            author.save()
        else :
            print(form.errors)
        return render(request,'core/spamsubmission.html',{'form':form,'predict':forcast })    


    else:
        form = ClassifierForm()
    return render(request,'core/spamsubmission.html',{'form':form, })    
