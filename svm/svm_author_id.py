#!/usr/bin/python

"""
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors

    Sara has label 0
    Chris has label 1

"""

import sys
from time import time
import os
dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append("../tools/")
# sys.path.append("../choose_your_own/")
from email_preprocess import preprocess
# from class_vis import prettyPicture

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn import svm
clf = svm.SVC(C=10000.0,kernel="rbf")

t0 = time()
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print(pred)
print "predicting time:", round(time()-t0, 3), "s"

# Count how many times it is Chris
print("total features:", len(features_test))
count = 0

for item in pred:
    if item == 1:
        count += 1

print("count:", count)

from sklearn.metrics import accuracy_score
# print(accuracy_score(labels_test, pred))
#########################################################
