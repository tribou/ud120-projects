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
clf = svm.SVC(kernel="linear")

# t0 = time()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
# prettyPicture(clf, features_test, labels_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(labels_test, pred))
# print "total time:", round(time()-t0, 3), "s"
#########################################################
