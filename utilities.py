from __future__ import division  # floating point division
import numpy as np
import math
from sklearn.metrics import log_loss

# map the season to a numeric feature representing
# winter, spring, summer, fall
def map_season(x):
    if (x > 11 or x < 3):
        x = 1
    elif (x > 2 and x < 6):
        x = 2
    elif (x > 5 and x < 9):
        x = 3
    elif (x > 8 and x < 12):
        x = 4
    return x

def is_residential(district):
	if ((district == "SOUTHERN") or (district == "CENTRAL") or (district == "BAYVIEW")):
		return 0
	else:
		return 1

def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0


# compute log loss
def geterror(ytest, predictions):
    return log_loss(ytest, predictions)