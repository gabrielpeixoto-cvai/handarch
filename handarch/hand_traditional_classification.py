#static gesture classification classes
import cv2 as cv
import numpy as np
import logging
from time import time
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import os

np.set_printoptions(threshold=np.inf)

class mlpClassifier:
    def __init__(self, path, name):
        self.model_path =  path
        self.model_name = name
        self.save_path = path + name + ".mod"
        self.clf = MLPClassifier(activation='relu', solver='adam', alpha=1e-5, hidden_layer_sizes=(100), random_state=1, verbose=True)

    def train(self, features, labels, CLASSES):

        trainData, testData, trainLabels, testLabels = train_test_split(features, labels, test_size=0.20, random_state=None)

        t0 = time()

        self.clf.fit(trainData, trainLabels)

        #print(self.clf.feature_importances_)
        print(self.clf.score(testData, testLabels))

        if not os.path.isdir(os.path.split(self.model_path)[0]):
            os.makedirs(os.path.split(self.model_path)[0])
        joblib.dump(self.clf, self.save_path)

        #test training

        y_pred = self.clf.predict(testData)
        print("done in %0.3fs" % (time() - t0))

        print(classification_report(testLabels, y_pred, target_names=CLASSES))

        print("Confusion Matrix")
        print(confusion_matrix(testLabels, y_pred, labels=range(len(CLASSES))))

    def gridSearchTrain(self, features, labels, CLASSES):
        print("Spliting dataset into 70% for training and 30% for testing")
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=None)
        #np.array(list(X_train), dtype=np.float)

        t0 = time()
        param_grid = {'activation': ['identity', 'logistic', 'tanh', 'relu'],'solver': ['lbfgs', 'sgd', 'adam'], 'hidden_layer_sizes':[(100), (200), (300), (50,50),(100,100)],}
        clf = GridSearchCV(MLPClassifier(random_state=None, verbose=True), param_grid)
        print("Training a MLP Classifier")
        #clf.fit(fds, labels)
        clf = clf.fit(X_train, y_train)
        print("done in %0.3fs" % (time() - t0))
        print("best parameters: ")
        print(clf.best_params_)

        print("Predicting gesture classes on the test set")
        t0 = time()

        if not os.path.isdir(os.path.split(self.model_path)[0]):
            os.makedirs(os.path.split(self.model_path)[0])
        joblib.dump(self.clf, self.save_path)

        y_pred = clf.predict(X_test)
        print("done in %0.3fs" % (time() - t0))

        print(classification_report(y_test, y_pred, target_names=CLASSES))

        print("Confusion Matrix")
        print(confusion_matrix(y_test, y_pred, labels=range(len(CLASSES))))

    def parameterTrain(self, features, labels, CLASSES, act, solv, hidden):
        print("Spliting dataset into 70% for training and 30% for testing")
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=None)
        #np.array(list(X_train), dtype=np.float)

        #parameters
        activation = ['identity', 'logistic', 'tanh', 'relu']
        solver = ['lbfgs', 'sgd', 'adam']
        hidden_layer_sizes = [(100), (200),(100,100)]


        t0 = time()
        clf = MLPClassifier(activation=activation[act], solver=solver[solv], alpha=1e-5, hidden_layer_sizes=hidden_layer_sizes[hidden], random_state=1, verbose=True)
        print("Training a MLP Classifier using the following parameters")
        print("Activation function", activation[act])
        print("Solver", solver[solv])
        print("hidden_layer_sizes", hidden_layer_sizes[hidden])
        #clf.fit(fds, labels)
        clf = clf.fit(X_train, y_train)
        print("done in %0.3fs" % (time() - t0))

        print("Predicting gesture classes on the test set")
        t0 = time()

        if not os.path.isdir(os.path.split(self.model_path)[0]):
            os.makedirs(os.path.split(self.model_path)[0])
        joblib.dump(self.clf, self.save_path)

        y_pred = clf.predict(X_test)
        print("done in %0.3fs" % (time() - t0))

        print(classification_report(y_test, y_pred, target_names=CLASSES))

        print("Confusion Matrix")
        print(confusion_matrix(y_test, y_pred, labels=range(len(CLASSES))))



    def inference(self, clf, feature, probabilities=True):
        feature = feature.reshape(1,-1)

        pred = clf.predict(feature)
        class_probabilities = clf.predict_proba(feature)

        return pred, class_probabilities

    def crossValidate(self, features, labels, CLASSES):
        scores = cross_val_score(self.clf, features, labels, cv=10, scoring='f1_macro')
        print(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        


    def evaluate(self, features, labels, CLASSES):

        y_pred = self.clf.predict(testData)
        #print("done in %0.3fs" % (time() - t0))

        print(classification_report(testLabels, y_pred, target_names=CLASSES))

        print("Confusion Matrix")
        print(confusion_matrix(y_test, y_pred, labels=range(len(CLASSES))))

class svmClassifier:
    def __init__(self, path=".", name="model"):
        self.model_path =  path
        self.model_name = name
        self.save_path = path + name + ".mod"
        self.clf = SVC(kernel='rbf', class_weight='balanced', probability=True,verbose=True)
        self.model = joblib.load("pcahogSvmModel.model")
        #self.model = joblib.load("hogSvmModel.model")

    def train(self, features, labels, CLASSES):
        print("Spliting dataset into 70% for training and 30% for testing")
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=None)
        #np.array(list(X_train), dtype=np.float)

        t0 = time()
        param_grid = {'C': [1e-2, 1e1, 1e2, 1e3, 1e4, 1e5],'gamma': [ 0.0001, 0.001, 0.01, 0.1,1], }
        clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced',probability=True, verbose=False), param_grid, n_jobs=2)
        print("Training a SVM Classifier")
        #clf.fit(fds, labels)
        clf = clf.fit(X_train, y_train)
        print("done in %0.3fs" % (time() - t0))
        print("best parameters: ")
        print(clf.best_params_)

        print("Predicting gesture classes on the test set")
        t0 = time()

        if not os.path.isdir(os.path.split(self.model_path)[0]):
            os.makedirs(os.path.split(self.model_path)[0])
        joblib.dump(self.clf, self.save_path)

        y_pred = clf.predict(X_test)
        print("done in %0.3fs" % (time() - t0))

        print(classification_report(y_test, y_pred, target_names=CLASSES))

        print("Confusion Matrix")
        print(confusion_matrix(y_test, y_pred, labels=range(len(CLASSES))))

    def trainParameter(self, features, labels, CLASSES, c_index, g_index):
        print("Spliting dataset into 70% for training and 30% for testing")
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=None)

        c = [1e-2, 1e1, 1e2, 1e3, 1e4, 1e5]
        gamma = [ 0.0001, 0.001, 0.01, 0.1,1]

        t0 = time()
        clf = SVC(kernel='rbf', C=c[c_index], gamma=gamma[g_index],class_weight='balanced',probability=True, verbose=False)
        print("Training a SVM Classifier")
        print("with parameters:")
        print("C: ", c[c_index])
        print("Gamma: ", gamma[g_index])
        #clf.fit(fds, labels)
        clf = clf.fit(X_train, y_train)
        print("done in %0.3fs" % (time() - t0))

        print("Predicting gesture classes on the test set")
        t0 = time()

        if not os.path.isdir(os.path.split(self.model_path)[0]):
            os.makedirs(os.path.split(self.model_path)[0])
        joblib.dump(self.clf, self.save_path)

        y_pred = clf.predict(X_test)
        print("done in %0.3fs" % (time() - t0))

        print(classification_report(y_test, y_pred, target_names=CLASSES))

        print("Confusion Matrix")
        print(confusion_matrix(y_test, y_pred, labels=range(len(CLASSES))))

    def inference(self, feature):
        t0 = time()
        res = self.model.predict(feature)
        print("prediction done in %0.3fs" % (time() - t0))
        return res


    def crossValidate(self, features, labels, CLASSES):
        scores = cross_val_score(self.clf, features, labels, cv=10, scoring='f1_macro')
        print(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

