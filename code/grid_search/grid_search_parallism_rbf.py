__author__ = 'pierregagliardi'

import cPickle
import time
import os
import sys
import multiprocessing
from itertools import product

import numpy as np
from sklearn.externals.joblib import Parallel, delayed
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from projet_sentiment_analysis.code.utilities import extract_data
# Train classifiers
def fit_and_score((C,gamma),X_train, y_train,X_test):
     start_time = time.clock()
     clf=SVC(C=C, cache_size=2000, class_weight=None,
            gamma=gamma, kernel='rbf', max_iter=-1, probability=False, shrinking=True,
            tol=0.001, verbose=False)
     #First the model is train on the training set
     clf.fit(X_train, y_train)
     #Then we test it on the testing set
     y_prediction=clf.predict(X_test)
     score=metrics.accuracy_score(y_test, y_prediction)
     end_time = time.clock()
     print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
     print("Parameters are %s, %s with a score of %0.2f"
     %(C, gamma,score ))
     return metrics.accuracy_score(y_test, y_prediction)


if __name__ == "__main__":


    ####### home computer path ######
    general_path='/Users/pierregagliardi/DossierTravail/Programmation/PythonPath/projet_sentiment_analysis/'
    path_to_training_set=general_path+'training_set_60000/training_set_unigram_all_features/'
    path_to_pickle=general_path+'pickle_hyper_parameters/'
    (X_train, y_train, X_test, y_test,number_training, number_testing)= extract_data.extract_training_and_testing_set(
        path_to_training_set+'metrics_training_set_1000.data',
        path_to_training_set+'metrics_testing_set_1000.data')


    #check how many cpu are available to parallelise the task
    print multiprocessing.cpu_count()

    # For an initial search, a logarithmic grid with basis
    # 10 is often helpful. Using a basis of 2, a finer
    # tuning can be achieved but at a much higher cost.

    C_range = np.logspace(0, 3, 4)
    gamma_range = np.logspace(-3, -1, 3)
    param_grid = dict(gamma=gamma_range, C=C_range)


    #paralellise the research of hyper parameters
    n_jobs=multiprocessing.cpu_count()-2

    params_scores = Parallel(
        n_jobs=n_jobs, verbose=True,
        )(
        delayed(fit_and_score)(param, X_train, y_train, X_test)
        for param in list(product(C_range,gamma_range))
        )


    #pickle the performance of the model
    with open(path_to_pickle+'60000_all_features_1000_rbf.pkl', 'wb') as fid :
            cPickle.dump(params_scores, fid)


