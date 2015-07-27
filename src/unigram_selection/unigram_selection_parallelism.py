__author__ = 'pierregagliardi'

import cPickle
import numpy as np
import multiprocessing
from sklearn.externals.joblib import Parallel, delayed
from sklearn import metrics
from sklearn import svm
from sklearn.feature_selection import VarianceThreshold

from projet_sentiment_analysis.code.utilities import extract_data

#This function finds the optimal number of unigram to use to classify tweets.
def unigram_selection(i,path_to_training_set,path_to_pickle):

 (X_train, y_train, X_test, y_test,number_training, number_testing)= extract_data.extract_training_and_testing_set(
    path_to_training_set+'metrics_training_set_%d.data'%i,
    path_to_training_set+'metrics_testing_set_%d.data'%i)

 print(X_train[0].__len__())
 clf=svm.SVC(C=1, cache_size=2000, class_weight=None, coef0=0.0, degree=3,
 gamma=0.1, kernel='linear', max_iter=-1, probability=False, shrinking=True,
 tol=0.001, verbose=False)

 clf.fit(X_train, y_train)


 print ()
 print("Detailed classification report:")
 print()
 print("The model is trained on the full development set: %d" % number_training)
 print("The scores are computed on the full evaluation set: %d" % number_testing)
 print()
 y_true=y_test
 y_prediction=clf.predict(X_test)

 print(metrics.classification_report(y_true, y_prediction))

 clf_metrics=np.vstack((y_true,y_prediction))

 with open(path_to_pickle+'60000_all_features_%d.pkl'%i, 'wb') as fid :
    cPickle.dump(clf_metrics, fid)
 print()

if __name__ == "__main__":

    general_path='/Users/pierregagliardi/DossierTravail/Programmation/PythonPath/projet_sentiment_analysis/'
    path_to_training_set=general_path+'training_set_60000/training_set_unigram_all_features/'
    path_to_pickle=general_path+'pickle_unigrams_without_varThres/'

    unigrams=[1000,2000,3000,4000,5000,6000,7000,8000,9000]
    #The jobs are also parallelised on the cpu
    n_jobs=multiprocessing.cpu_count()-2

    params_scores = Parallel(
        n_jobs=n_jobs, verbose=True,
        )(
        delayed(unigram_selection)(i,path_to_training_set,path_to_pickle)
        for i in unigrams)
