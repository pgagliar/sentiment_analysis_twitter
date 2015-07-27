__author__ = 'pierregagliardi'

import cPickle
import time
import os
import sys
from itertools import product

import numpy as np
from sklearn.externals.joblib import Parallel, delayed
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold

from projet_sentiment_analysis.code.utilities import extract_data

def fit_and_score(C,X_train, y_train,X_test):
     start_time = time.clock()
     clf=SVC(C=C, cache_size=2000, class_weight=None,
             kernel='linear', max_iter=-1, probability=False, shrinking=True,
             tol=0.001, verbose=False)
     clf.fit(X_train, y_train)
     y_prediction=clf.predict(X_test)
     score=metrics.accuracy_score(y_test, y_prediction)
     end_time = time.clock()
     print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
     print("Parameter is %s with a score of %0.2f"
     %(C,score ))
     return metrics.accuracy_score(y_test, y_prediction)

# Utility function to move the midpoint of a colormap to be around
# the values of interest.
if __name__ == "__main__":

    ####### home computer path ######
    general_path='/Users/pierregagliardi/DossierTravail/Programmation/PythonPath/projet_sentiment_analysis/'
    #path_to_training_set=general_path+'training_set_random/'
    path_to_training_set=general_path+'training_set_60000/training_set_unigram_all_features/'
    path_to_pickle=general_path+'pickle_hyper_parameters/'

    (X_train, y_train, X_test, y_test,number_training, number_testing)= extract_data.extract_training_and_testing_set(
        path_to_training_set+'metrics_training_set_1000.data',
        path_to_training_set+'metrics_testing_set_1000.data')


    ##############################################################################
    # Train classifiers
    #
    # For an initial search, a logarithmic grid with basis
    # 10 is often helpful. Using a basis of 2, a finer
    # tuning can be achieved but at a much higher cost.

    C_range = np.logspace(-1, 3, 5)


    params_scores = Parallel(
        n_jobs=3, verbose=True,
        )(
        delayed(fit_and_score)(param, X_train, y_train, X_test)
        for param in C_range
    )


    with open(path_to_pickle+'60000_all_features_1000_linear_without_varThres.pkl', 'wb') as fid :
            cPickle.dump(params_scores, fid)

