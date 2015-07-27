from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC


def extract_training_and_testing_set(path_training, path_testing):
    y_train = []
    y_test = []
    X_train = []
    X_test = []
    i_train=0
    i_test=0
    for line in open(path_training, "rU"):
        temp_train = line.split(',')
        y_train.append(temp_train[0])
        X_train.append(map(int, temp_train[1:]))
        i_train+=1

    for line in open(path_testing, "rU"):
        temp_test = line.split(',')
        y_test.append(temp_test[0])
        X_test.append(map(int, temp_test[1:]))
        i_test+=1
    return (X_train, y_train, X_test, y_test,i_train,i_test)


if __name__ == "__main__":

 (X_train, y_train, X_test, y_test,number_training, number_testing) = extract_training_and_testing_set('metrics_training_set.data','metrics_testing_set.data')
 estimator = SVC(C=1.0, cache_size=3000, class_weight=None, coef0=0.0, degree=3,
                gamma=0.0, kernel='rbf', max_iter=-1, probability=False, random_state=None,
                shrinking=True, tol=0.001, verbose=False)  # Set the parameters by cross-validation
 tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

 scores = ['precision', 'recall']

 for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(estimator, tuned_parameters, cv=5, scoring=score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set: %d" % number_training)
    print("The scores are computed on the full evaluation set: %d" % number_testing)
    print()
    y_true=y_test
    y_prediction=clf.predict(X_test)
    print(classification_report(y_true, y_prediction))
    print()