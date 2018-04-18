# -*- coding: utf-8 -*-
"""
Created on Thu Dec 07 23:51:06 2017

@author: Durgaraju
"""

 #get some libraries that will be useful
import re
#import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# the Naive Bayes model
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
# function to split the data for cross-validation
from sklearn.model_selection import train_test_split
# function for transforming documents into counts
from sklearn.feature_extraction.text import CountVectorizer
# function for encoding categories
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

import seaborn as sb


# grab the data
tweets = pd.read_csv("data.csv")
tweets.head()


def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist and x != 'the']
    return ulist

def normalize_text(s):
    s = s.lower()
    
    #Convert www.* or https?://* to URL
    s = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',s)
    #Convert @username to AT_USER
    s = re.sub('@[^\s]+',' ',s)
    #Replace #word with word
    s = re.sub(r'#([^\s]+)', r'\1', s)
    #trim
    s = s.strip('\'"')
    
    # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)
    s = re.sub('\W',' ',s)
    s = re.sub('\W',' ',s)
    
    # make sure we didn't introduce any double spaces
    s = re.sub('\s+',' ',s)
    
    s = ''.join([i for i in s if not i.isdigit()])
    
    s=' '.join(unique_list(s.split()))
    
    s = [e.lower() for e in s.split() if len(e) >= 3] 
    s = ' '.join(s)
    return s

tweets['NTweet'] = [normalize_text(s) for s in tweets['Tweet']]


# pull the data into vectors

vector = CountVectorizer()
x = vector.fit_transform(tweets['NTweet'])

encode = LabelEncoder()
y = encode.fit_transform(tweets['Sentiment'])


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state = 10)
    
c = BernoulliNB()

c.fit(x_train, y_train)
    
y_pred = c.predict(x_test)

   
print("Bernoulli Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)

a = confusion_matrix(y_test, y_pred)


b = pd.DataFrame(a, index = [i for i in np.unique(y)],
                  columns = [i for i in np.unique(y)])

plt.figure(figsize = (5,5))

sb.heatmap(b, annot=True)

# Making the Confusion Matrix
x = accuracy_score(y_test,y_pred)
print("Accuracy is",x)

