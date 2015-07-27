__author__ = 'pierregagliardi'

import numpy as np
import pickle
from sklearn.feature_selection import VarianceThreshold

from projet_sentiment_analysis.code.utilities import extract_data


if __name__ == "__main__":

    general_path='/Users/pierregagliardi/DossierTravail/Programmation/PythonPath/projet_sentiment_analysis/'
    path_to_training_set=general_path+'training_set_60000/training_set_unigram_all_features/'
    path_to_pickle=general_path+'pickle_hyper_parameters/'

    (X_train, y_train, X_test, y_test,number_training, number_testing)= extract_data.extract_training_and_testing_set(
        path_to_training_set+'metrics_training_set_7000.data',
        path_to_training_set+'metrics_testing_set_7000.data')

    sel = VarianceThreshold(threshold=(.999* (1 - .999)))
    X_train=sel.fit_transform(X_train)
    X_test=sel.transform(X_test)

    with open(path_to_pickle+'metrics_60000_all_features_7000.pkl', 'wb') as fid :
            pickle.dump((X_train, y_train, X_test, y_test), fid)
