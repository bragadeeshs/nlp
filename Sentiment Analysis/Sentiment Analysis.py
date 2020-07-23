#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.python.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import bz2
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import re

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Retrieve the data and label from the dataset

# In[2]:


def get_labels_and_texts(file):
    labels = []
    texts = []
    for line in bz2.BZ2File(file):
        x = line.decode("utf-8")
        labels.append(int(x[9]) - 1)
        texts.append(x[10:].strip())
    return np.array(labels), texts
train_labels, train_texts = get_labels_and_texts('train.ft.txt.bz2')
test_labels, test_texts = get_labels_and_texts('test.ft.txt.bz2')


# ## Subset of the data to train

# In[155]:


train_labels=train_labels[0:2000]
train_texts=train_texts[0:2000]


# ## Use Regular Expression to process the data

# In[156]:


import re
NON_ALPHANUM = re.compile(r'[\W]')
NON_ASCII = re.compile(r'[^a-z0-1\s]')
def normalize_texts(texts):
    normalized_texts = []
    for text in texts:
        lower = text.lower()
        no_punctuation = NON_ALPHANUM.sub(r' ', lower)
        no_non_ascii = NON_ASCII.sub(r'', no_punctuation)
        normalized_texts.append(no_non_ascii)
    return normalized_texts
        
train_texts = normalize_texts(train_texts)
test_texts = normalize_texts(test_texts)


# ## Create a sparse matrix using Count Vectorizer

# In[157]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(binary=True)
cv.fit(train_texts)
X = cv.transform(train_texts)
X_test = cv.transform(test_texts)


# ## Deploy Classification Models and find the best combination for the model based on the accuracy

# In[158]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


X_train, X_val, y_train, y_val = train_test_split(
    X, train_labels, train_size = 0.75
)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_val, lr.predict(X_val))))


# In[159]:


from sklearn.neighbors import KNeighborsClassifier

for k in range(2,8):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    print ("Accuracy for K=%s: %s" 
           % (k, accuracy_score(y_val, knn.predict(X_val))))


# In[160]:


from sklearn import svm
for i in [0.001,0.01,0.1,1,10,1000]:
    clf = svm.SVC(probability=True, C = i)
    clf.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (i, accuracy_score(y_val, clf.predict(X_val))))    


# In[161]:


from sklearn.tree import DecisionTreeClassifier

for i in ['gini', 'entropy']:
    for j in [2,3,4,5]:
        dt = DecisionTreeClassifier(criterion = i, max_depth = j)
        dt.fit(X_train, y_train)
        print ("Accuracy for Criterion=%s ,Max_Depth = %i: %s" 
               % (i, j, accuracy_score(y_val, dt.predict(X_val))))             


# In[162]:


from sklearn.ensemble import RandomForestClassifier

for i in ['gini', 'entropy']:
    for j in [2,3,4,5]:
        rf = RandomForestClassifier(criterion = i, max_depth = j)
        rf.fit(X_train, y_train)
        print ("Accuracy for Criterion=%s ,Max_Depth = %i: %s" 
               % (i, j, accuracy_score(y_val, rf.predict(X_val))))   


# In[163]:


from sklearn.ensemble import BaggingClassifier

for i in [2,3,4,5,6,7]:
    bg = BaggingClassifier(n_estimators=i)
    bg.fit(X_train, y_train)
    print ("Accuracy for Estimators=%s: %s" 
           % (i, accuracy_score(y_val, bg.predict(X_val))))  


# In[164]:


from sklearn.ensemble import AdaBoostClassifier

for i in [0.001,0.01,0.1,1,10]:
    ad = AdaBoostClassifier(learning_rate=i)
    ad.fit(X_train, y_train)
    print ("Accuracy for LearningRate=%s: %s" 
           % (i, accuracy_score(y_val, ad.predict(X_val))))  


# ## Create a review and predict the review for few classifiers

# In[180]:


review = ['The product is very worst, please buy']
review_normalized = normalize_texts(review)
review_normalized_test = cv.transform(review_normalized)
review_normalized_test


# In[181]:


bg = BaggingClassifier(n_estimators=6)
bg.fit(X_train, y_train)
bg.predict(review_normalized_test)


# In[182]:



clf = svm.SVC(probability=True, C = 10)
clf.fit(X_train, y_train)
clf.predict(review_normalized_test)    


# In[ ]:




