from __future__ import division  # floating point division
import numpy as np
import random as rand
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sknn.mlp import Classifier as clf_neural
from sknn.ae import AutoEncoder
from sknn import ae
from sknn.mlp import Layer
from sknn.ae import AutoEncoder as ae
from sknn.backend import lasagne
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import make_scorer, log_loss
import xgboost as xgb
import utilities as utils

class Classifier:
    
    def __init__( self, params=None ):
        """ implement this """

    def learn(self, Xtrain, ytrain):
        """ implement this """

    def predict(self, Xtest):
        """ implement this """

# Bernoulli Naive Bayes
# log loss score : ~2.55, Time : ~1.6 seconds   
class NaiveBayes(Classifier):
    
    def __init__( self, params=None ):

        self.clf = BernoulliNB()
 
    def learn(self, Xtrain, ytrain):
        self.clf.fit(Xtrain, ytrain)

    def predict(self, Xtest, encoder):
        probs = self.clf.predict_proba(Xtest)
        return probs

# Logistic Regression, Not Bad
# log loss score : ~2.53, Time : ~262 seconds
class LogitReg(Classifier):
    
    def __init__( self, params=None ):

        self.clf = LogisticRegression()

    def learn(self, Xtrain, ytrain):
        print('Fitting Model...')
        self.clf.fit(Xtrain, ytrain)

    def predict(self, Xtest, encoder):
        print('Making Predictions...')
        probs = self.clf.predict_proba(Xtest)
        return probs

# 4 layer Neural Network
# Used this for competition, currently at ~2.37, Time : ~155 seconds
class NeuralNetwork(Classifier):
    
    def __init__( self, params=None ):

        self.clf = clf_neural(layers=[Layer("Tanh", units=128), Layer("Tanh", units=128),Layer("Sigmoid", units=128),
         Layer("Softmax")], learning_rate=0.08 ,batch_size=100, n_iter=500, dropout_rate=0.4,
         learning_rule="adagrad")

        
        self.scaler = StandardScaler()

    def learn(self, Xtrain, ytrain):
        xtrain = self.scaler.fit_transform(Xtrain)
        print('Fitting Model...')
        self.clf.fit(np.array(xtrain), np.array(ytrain))

    def predict(self, Xtest, encoder):
        xtest = self.scaler.transform(Xtest)
        print('Making Predictions...')
        probs = self.clf.predict_proba(np.array(xtest))
        # print('Writing data to csv...')
        # actuals = pd.DataFrame(probs, columns=encoder.classes_)
        # actuals.to_csv('neural_results.csv', index=True, index_label = 'Id')
        return probs


class XGB(Classifier):

    def __init__( self, params=None ):
        
        self.params = {'max_depth': 8, 'eta':0.1, 'silent':1,
              'objective':'multi:softprob', 'num_class':39, 'eval_metric':'mlogloss',
              'min_child_weight':3, 'subsample':0.5,'colsample_bytree':0.5 }
        
        self.xgb_class = None
        
    def learn(self, Xtrain, ytrain):
        xgb_train = xgb.DMatrix(Xtrain, ytrain)
        print('Fitting Model...')
        self.xgb_class = xgb.train(self.params, xgb_train, 275)

    def predict(self, Xtest, encoder):
        print('Making Predictions...')
        probs = self.xgb_class.predict(xgb.DMatrix(Xtest))
        # print('Writing data to csv...')
        # actuals = pd.DataFrame(probs, columns=encoder.classes_)
        # actuals.to_csv('xgb_test.csv', index=True, index_label = 'Id')
        return probs


# Random Forest Classifier
# log loss score : ~2.57, Time : ~25 seconds, Features: pd, day of week, intersection
class RFC(Classifier):
    
    def __init__( self, params=None ):
        
        self.clf = RandomForestClassifier(n_estimators=10, criterion="entropy")

    def learn(self, Xtrain, ytrain):
        self.clf.fit(Xtrain, ytrain)

    def predict(self, Xtest, encoder):
        probs = self.clf.predict_proba(Xtest)        
        return probs
