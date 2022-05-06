import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.svm import SVC
import os

target = []
images = []
flat_data = []

datadir = 'Fish_Dataset'
i=1
categories = ['Black Sea Sprat', 'Gilt-Head Bream', 'Hourse Mackerel', 'Red Mullet', 'Red Sea Bream', 'Sea Bass','Shrimp', 'Striped Red Mullet', 'Trout']
for category in categories:
    class_num = categories.index(category)
    path = os.path.join(datadir, category)
    for img in os.listdir(path):
        print(i)
        img_array = imread(os.path.join(path, img))
        img_resized = resize(img_array, (150, 150, 3))
        flat_data.append(img_resized.flatten())
        images.append(img_resized)
        target.append(class_num)
        i+=1
print('images converted to array')
flat_data = np.array(flat_data)
target = np.array(target)
images = np.array(images)
print("images converted")
# split data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(flat_data, target, test_size=0.3)
print(x_train.shape)

# # Logistic Regression
# logistic = LogisticRegression()
# print("training logistic regression model")
# logistic.fit(x_train, y_train)
# print("model trained")
# # MODEL EVALUATION
# # ACCURACY SCORE
# xtest_pred = logistic.predict(x_test)
# test_score = accuracy_score(xtest_pred, y_test)
# print('Accuracy: ', round(test_score * 100, 2))
# cm = confusion_matrix(y_test, xtest_pred)
# print(cm)
# # PLOT CONFUSION MATRIX
# plt.matshow(cm)
# plt.title('Confusion Matrix')
# plt.colorbar()
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()
# plt.savefig('confusion_matrix_logistic.png')
#
# #DECISION TREE ALGORITHM
# from sklearn.tree import DecisionTreeClassifier
# decision = DecisionTreeClassifier()
# print("training decision tree classifier model")
# decision.fit(x_train, y_train)
# print("model trained")
# # MODEL EVALUATION
# # ACCURACY SCORE
# xtest_pred = decision.predict(x_test)
# test_score = accuracy_score(xtest_pred, y_test)
# print('Accuracy: ', round(test_score * 100, 2))
# cm = confusion_matrix(y_test, xtest_pred)
# print(cm)
# # PLOT CONFUSION MATRIX
# plt.matshow(cm)
# plt.title('Confusion Matrix')
# plt.colorbar()
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()
# plt.savefig('decision_tree.png')
#
#
# # Gaussian Naive Bayes classifier
# gaussian = GaussianNB()
# print("training gaussian model")
# gaussian.fit(x_train, y_train)
# print("model trained")
# # MODEL EVALUATION
# # ACCURACY SCORE
# xtest_pred = gaussian.predict(x_test)
# test_score = accuracy_score(xtest_pred, y_test)
# print('Accuracy: ', round(test_score * 100, 2))
# cm = confusion_matrix(y_test, xtest_pred)
# print(cm)
# # PLOT CONFUSION MATRIX
# plt.matshow(cm)
# plt.title('Confusion Matrix')
# plt.colorbar()
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()
# plt.savefig('confusion_matrix_naivegauss.png')
#
# # RANDOM FOREST CLASSIFIER
# randomforest = RandomForestClassifier(max_depth=100, random_state=0)
# print("training random forest model")
# randomforest.fit(x_train, y_train)
# print("model trained")
# # MODEL EVALUATION
# # ACCURACY SCORE
# xtest_pred = randomforest.predict(x_test)
# test_score = accuracy_score(xtest_pred, y_test)
# print('Accuracy: ', round(test_score * 100, 2))
# cm = confusion_matrix(y_test, xtest_pred)
# print(cm)
# # PLOT CONFUSION MATRIX
# plt.matshow(cm)
# plt.title('Confusion Matrix')
# plt.colorbar()
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()
# plt.savefig('confusion_matrix_randomforest.png')
# # KNEARESTNEIGHBORS CLASSIFIER
# knearest = KNeighborsClassifier(n_neighbors=x_train.shape[0])
# print("training knn model")
# knearest.fit(x_train, y_train)
# print("model trained")
# # MODEL EVALUATION
# # ACCURACY SCORE
# xtest_pred = knearest.predict(x_test)
# test_score = accuracy_score(xtest_pred, y_test)
# print('Accuracy: ', round(test_score * 100, 2))
# cm = confusion_matrix(y_test, xtest_pred)
# print(cm)
# # PLOT CONFUSION MATRIX
# plt.matshow(cm)
# plt.title('Confusion Matrix')
# plt.colorbar()
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()
# plt.savefig('confusion_matrix_knn.png')

#STOCASTIC GRADIENT DESCENT
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss='log', penalty='l2', max_iter=1000)
print("training sgd model")
sgd.fit(x_train, y_train)
print("model trained")
# MODEL EVALUATION
# ACCURACY SCORE
xtest_pred = sgd.predict(x_test)
test_score = accuracy_score(xtest_pred, y_test)
print('Accuracy: ', round(test_score * 100, 2))
cm = confusion_matrix(y_test, xtest_pred)
print(cm)
# PLOT CONFUSION MATRIX
plt.matshow(cm)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
plt.savefig('confusion_matrix_sgd.png')

#SVM CLASSIFIER
# svm = SVC(gamma='auto')
# print("training svm model")
# svm.fit(x_train, y_train)
# print("model trained")
# # MODEL EVALUATION
# # ACCURACY SCORE
# xtest_pred = svm.predict(x_test)
# test_score = accuracy_score(xtest_pred, y_test)
# print('Accuracy: ', round(test_score * 100, 2))
# cm = confusion_matrix(y_test, xtest_pred)
# print(cm)
# # PLOT CONFUSION MATRIX
# plt.matshow(cm)
# plt.title('Confusion Matrix')
# plt.colorbar()
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()
# plt.savefig('confusion_matrix_svm.png')
#
# #CUSTOM INPUT
# flatdata=[]
# predictions=[]
# custom=input("Enter your own input: ")
# img_array = imread(custom)
# img_resized = resize(img_array, (150, 150, 3))
# flatdata.append(img_resized.flatten())
# flatdata = np.array(flatdata)
# predictions.append(randomforest.predict(flatdata))
# predictions.append(logistic.predict(flatdata))
# predictions.append(knearest.predict(flatdata))
# predictions.append(gaussian.predict(flatdata))
# predictions.append(decision.predict(flatdata))
# print(predictions)