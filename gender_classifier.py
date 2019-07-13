from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split


X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


#Split the the data into testing and training data
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.3, random_state=4)

#Decision Tree
clf_tree = tree.DecisionTreeClassifier()
clf_tree.fit(X_train, Y_train)
prediction_tree = clf_tree.predict(X_test)
accuracy_tree = metrics.accuracy_score(Y_test, prediction_tree)

#kNN

clf_knn = KNeighborsClassifier(n_neighbors = 1).fit(X_train,Y_train)
prediction_knn = clf_knn.predict(X_test)
accuracy_knn = metrics.accuracy_score(Y_test, prediction_knn)

#SVM
clf_svm = svm.SVC(kernel='rbf', gamma='scale')
clf_svm.fit(X_train, Y_train)
prediction_svm = clf_svm.predict(X_test)
accuracy_svm = metrics.accuracy_score(Y_test, prediction_svm)

#Logistic Regression
clf_lr = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,Y_train)
prediction_lr = clf_lr.predict(X_test)
accuracy_lr = metrics.accuracy_score(Y_test, prediction_lr)

#Displays classifier with the highest accuracy
accuracy = {'Decision Tree':accuracy_tree,'kNN':accuracy_knn,'SVM':accuracy_svm,'Logistic Regression':accuracy_lr}
best_classifier = max(accuracy, key=accuracy.get)
print('The best classifier is ', best_classifier)


if (best_classifier == 'Decision Tree'):
    print ('The prediction of measurements [190, 70, 43] with ',best_classifier,'is',clf_tree.predict([[190, 70, 43]]))
elif (best_classifier == 'kNN'):
    print ('The prediction of measurements [190, 70, 43] with ',best_classifier,'is',clf_knn.predict([[190, 70, 43]]))
elif (best_classifier == 'SVM'):
    print ('The prediction of measurements [190, 70, 43] with ',best_classifier,'is',clf_svm.predict([[190, 70, 43]]))
elif (best_classifier == 'Logistic Regression'):
    print ('The prediction of measurements [190, 70, 43] with ',best_classifier,'is',clf_lr.predict([[190, 70, 43]]))
