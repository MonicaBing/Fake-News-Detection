#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detecting fake news 

tf = #t in doc/ total words in doc 
idf = log(total words in doc/ #t in doc
         >tfidf_vectorizer - return the tf-idf feature for the text
    

PassiveAggresiveClassifier - online-learning alog
>generally used for large-sclae learning 
>very useful where there is a hufe amount ofd data and it is computationally infeasible to traing the entire data set 
> good e.g.-> fake news on social media 
> dynamically read data from social media continuously, the data would be huge, using online-learning alog would be ideal 


Confusion matrix
> used to identify TP, TN, FP, FN
> used to define the performance of classification alogirthm

"""

import numpy as np 
import pandas as pd
import itertools #creating efficient loops and iterating 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer #doc said sklearn.feature_extraction, needed upadte 
#read the data ------------------------------

df=pd.read_csv('news.csv')
df.shape
df.head()

labels=df.label
labels.head()

#training/ testing --------------------------

#test size = proportion of test samples, random state = random shuffle, with the int allow regeneratino 
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)

#initilize a TfidfVectorizer-----------------


#stop_words = only allow english at this point, max_df = anything more than 0.7 will be discarded
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)


tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

#inirilize PassiveAggressiveClassifier ------------------

pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

y_pred=pac.predict(tfidf_test) 

score = accuracy_score(y_test, y_pred)

print(f'Accuracy: {round(score*100,2)}%')

#confusion matrix - performance check-------------------
confusion_matrix(y_test, y_pred, labels=['FAKE','REAL']) #labels != labels define before, parameters for this function 


#result: 589 TP, 578 TN, 42 FP, 49 FN

