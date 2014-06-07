from __future__ import division

import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.grid_search import GridSearchCV

trainFile = "train.csv"
testFile = "test.csv"
submissionFile = "extra_tree_submission2.csv"

dfTrain = pd.read_csv(trainFile)
dfTest = pd.read_csv(testFile)

features = [col for col in dfTrain.columns if col not in ['Cover_Type','Id']]

xTrainFull = dfTrain[features]
xTestFull = dfTest[features]
yTrainFull = dfTrain['Cover_Type']

"""
the best performing classifier was an extremely random forest, which outperformed kNN classifiers and SVC, including a variety of weighting and dimensionality reduction tests.

use 12k estimators for final submission using parameters from the grid search in the exploration notebook. 
"""
exTree = ensemble.ExtraTreesClassifier(n_estimators = 12000, 
                                       max_features=11,
                                       min_samples_split=1,
                                       n_jobs = 1,
                                       random_state=10)
print "training ensemble..."
exTree.fit(xTrainFull,yTrainFull)
print "done training."
print "predicting test data..."
predictions = exTree.predict(xTestFull)
print "done prediction."

# export to csv
outfile = open(submissionFile, 'w')
outfile.write('Id,Cover_Type\n')
for i in xrange(len(predictions)):
    outfile.write('%d,%d\n' % (dfTest['Id'][i], predictions[i]))
outfile.close()