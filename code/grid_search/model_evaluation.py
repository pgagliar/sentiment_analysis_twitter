__author__ = 'pierregagliardi'

import time
import os
import sys
from sklearn import metrics
from sklearn.svm import SVC
from projet_sentiment_analysis.code.utilities import extract_data


#Evaluation of the model on the test set
if __name__ == "__main__":


    general_path='projet_sentiment_analysis/'
    path_to_training_set=general_path+'training_set_60000/training_set_unigram_all_features/'
    path_to_pickle=general_path+'pickle_hyper_parameters/'

    (X_train, y_train, X_test, y_test,number_training, number_testing)= extract_data.extract_training_and_testing_set(
        path_to_training_set+'metrics_training_set_7000.data',
        path_to_training_set+'metrics_testing_set_7000.data')


    start_time = time.clock()
    clf=SVC(C=100, cache_size=2000, class_weight=None,gamma=0.01,
        kernel='rbf', max_iter=-1, probability=False, shrinking=True,
        tol=0.001, verbose=False)
    clf.fit(X_train, y_train)
    y_prediction=clf.predict(X_test)
    print(metrics.classification_report(y_test, y_prediction))
    end_time = time.clock()
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))




