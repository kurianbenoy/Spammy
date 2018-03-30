import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk import word_tokenize, NaiveBayesClassifier, classify, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix, accuracy_score

#reading data from CSV file
df = pd.read_csv('spam_ham.csv', encoding = "ISO-8859-1", skipinitialspace=True)
#renaming and shuffling the columns
df.columns = ['label','text']
df = df[['text', 'label']]

# print(df.head(),df.info())

#creating the bag of words model
CommonWords = stopwords.words('english')
wordLemmatizer = WordNetLemmatizer()
corpus = []
for i in range(0, 5572):
    #reading each text
    text = df['text'][i]
    #lemmatizing each word of the text. When we tokeninze a sentence we get individual words 
    wordtokens = [wordLemmatizer.lemmatize(word.lower()) for word in word_tokenize(text)] 
    #filtering out the stopwords from the text and combining them into a list again.
    text = ' '.join([x for x in wordtokens if x not in CommonWords])
    corpus.append(text)

# print(corpus[0:3])

from sklearn.feature_extraction.text import CountVectorizer
#creating the sparse matrix
cv= CountVectorizer(max_features= 5000)  
X= cv.fit_transform(corpus).toarray()
y= df.iloc[:,1].values
print(X,y)


#Splitting Dataset into train to test dataset
from sklearn.model_selection import train_test_split
print(X.shape,y.shape)
X = X.reshape(9194,)
print(X.shape,y.shape)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# from sklearn.naive_bayes import GaussianNB
# classifierNB = GaussianNB()
# classifierNB.fit(X_train, y_train)
# #making the prediction for the test results
# y_pred_NB = classifierNB.predict(X_test)
# print(y_pred_NB,y_test)

# #making the confusion matrix
# cm_NB = confusion_matrix(y_test, y_pred_NB)
# print(cm_NB)

