import random
import math
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import classalgorithms as algs
import time
import utilities as utils

# load the test data that produces our submission results
def loadcrimedata_test():
    
    file_path = '../data/test.csv'

    dataset = pd.read_csv(file_path,parse_dates=['Dates'])

    # this is for the testset
    dummy_days = pd.get_dummies(dataset.DayOfWeek)
    dummy_dis = pd.get_dummies(dataset.PdDistrict)
    hour = dataset.Dates.dt.hour
    month = dataset.Dates.dt.month
    year = dataset.Dates.dt.year
    mean_x = dataset.X.mean()
    mean_y = dataset.Y.mean()
    testdata = pd.concat([dummy_days, dummy_dis], axis=1)
    testdata['Month'] = month
    testdata['Hour'] = hour
    testdata['Year'] = year
    testdata['X'] = dataset.X.map(lambda x : mean_x if abs(x) < 121 else x)
    testdata['Y'] = dataset.Y.map(lambda x : mean_y if abs(x) > 40 else x)
    testdata['Night'] = dataset.Dates.dt.hour.map(lambda x : 1 if (x > 22 or x < 6) else 0)
    testdata['Intersection'] = dataset.Address.map(lambda x: 1 if '/' in x else 0)
    testdata['Season'] = dataset.Dates.dt.month.map(lambda x: utils.map_season(x))
    testdata['StreetNo'] = dataset['Address'].map(lambda x: x.split(' ', 1)[0] if x.split(' ', 1)[0].isdigit() else 0)

    
    return testdata
    
# load in the data and then split it accordingly, train and validation
def loadcrimedata():
    

    print('Loading data...')
    file_path = '../data/train.csv'

    dataset = pd.read_csv(file_path,parse_dates=['Dates'])

    # drop data that is in the north pole, wat?
    dataset = dataset[abs(dataset['Y']) < 50]
    
    label_en = preprocessing.LabelEncoder()
    
    category = label_en.fit_transform(dataset.Category)
    
    dummy_days = pd.get_dummies(dataset.DayOfWeek)
    dummy_dis = pd.get_dummies(dataset.PdDistrict)
    hour = dataset.Dates.dt.hour
    year = dataset.Dates.dt.year
    month = dataset.Dates.dt.month
    traindata = pd.concat([dummy_days, dummy_dis], axis=1)
    traindata['Month'] = month
    traindata['Hour'] = hour
    traindata['Year'] = year
    traindata['Category'] = category
    traindata['X'] = dataset.X
    traindata['Y'] = dataset.Y
    traindata['Night'] = dataset.Dates.dt.hour.map(lambda x : 1 if (x > 22 or x < 8) else 0)
    traindata['Intersection'] = dataset.Address.map(lambda x: 1 if '/' in x else 0)
    traindata['Season'] = dataset.Dates.dt.month.map(lambda x: utils.map_season(x))
    traindata['StreetNo'] = dataset['Address'].map(lambda x: x.split(' ', 1)[0] if x.split(' ', 1)[0].isdigit() else 0)
    
    trainset, testset = train_test_split(traindata, train_size=0.70)
    ytrain = trainset.Category
    ytest = testset.Category
    trainset = trainset.drop(['Category'], axis=1)
    testset= testset.drop(['Category'], axis=1)
    
    return (trainset, ytrain), (testset, ytest), label_en

if __name__ == '__main__':
    trainset, testset, encoder = loadcrimedata()
    testset_sub = loadcrimedata_test()
    print('Running on train={0} and test={1} samples').format(trainset[0].shape[0], testset[0].shape[0])
    classalgs = { 'Neural Network' : algs.NeuralNetwork() }
    # classalgs = { 'Random Forest' : algs.RFC() }
    # classalgs = {'Logistic Regression' : algs.LogitReg() }
    # classalgs = {'Naive Bayes' : algs.NaiveBayes() }
    # classalgs = { 'XGB Classifier' : algs.XGB() }
    for learnername, learner in classalgs.iteritems():
        print 'Running learner = ' + learnername
        t0 = time.time()
        learner.learn(np.array(trainset[0]), np.array(trainset[1]))
        t1 = time.time()
        print('Training time for ' + learnername + ': ' + str(t1 - t0) + ' seconds')
        # print(grid_scores)
        # print(best_params)
        # actual Test model
        probs = learner.predict(testset[0], encoder)
        # probs = learner.predict(np.array(testset_sub), encoder)
        # predictions = learner.predict(validationset[0], encoder)
        accuracy = utils.geterror(testset[1], probs)
        print 'Log loss for ' + learnername + ': ' + str(accuracy)
        # print('DONE WRITING TO CSV')
