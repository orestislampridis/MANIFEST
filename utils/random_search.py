from pprint import pprint

import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

n_jobs = -1


def get_knn_random_grid(x_train, y_train):
    # Create the parameter grid
    n_neighbors = [int(x) for x in np.linspace(start=1, stop=100, num=70)]

    param_grid = {'n_neighbors': n_neighbors}

    # Create a base model
    knnc = KNeighborsClassifier()

    # Manually create the splits in CV in order to be able to fix a random_state
    cv_sets = ShuffleSplit(n_splits=3, test_size=.33, random_state=42)

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=knnc,
                               param_grid=param_grid,
                               scoring='accuracy',
                               cv=cv_sets,
                               verbose=1,
                               n_jobs=n_jobs)

    # Fit the grid search to the data
    grid_search.fit(x_train, y_train)

    print("The best hyperparameters from Grid Search are:")
    print(grid_search.best_params_)
    print("")
    print("The mean accuracy of a model with these hyperparameters is:")
    print(grid_search.best_score_)

    best_knn = grid_search.best_estimator_
    print(best_knn)
    return best_knn


def get_NB_random_grid(x_train, y_train):
    # alpha
    alpha = [0, 1]

    # fit_prior
    fit_prior = [True, False]

    # Create the random grid
    random_grid = {'alpha': alpha,
                   'fit_prior': fit_prior}

    pprint(random_grid)

    # First create the base model to tune
    nbc = MultinomialNB()

    # Definition of the random search
    random_search = RandomizedSearchCV(estimator=nbc,
                                       param_distributions=random_grid,
                                       n_iter=50,
                                       scoring='accuracy',
                                       cv=3,
                                       verbose=1,
                                       random_state=42,
                                       n_jobs=n_jobs)

    # Fit the random search model
    random_search.fit(x_train, y_train)

    print("The best hyperparameters from Random Search are:")
    print(random_search.best_params_)
    print("")
    print("The mean accuracy of a model with these hyperparameters is:")
    print(random_search.best_score_)

    best_nb = random_search.best_estimator_
    print(best_nb)
    return best_nb


def get_logistic_regression_random_grid(x_train, y_train):
    # C
    C = [float(x) for x in np.linspace(start=0.1, stop=1, num=10)]

    # multi_class
    multi_class = ['ovr']

    # solver
    solver = ['liblinear', 'newton-cg', 'sag', 'saga', 'lbfgs']

    # penalty
    penalty = ['l2']

    # Create the random grid
    random_grid = {'C': C,
                   'multi_class': multi_class,
                   'solver': solver,
                   'penalty': penalty}

    pprint(random_grid)

    # First create the base model to tune
    lrc = LogisticRegression(random_state=42)

    # Definition of the random search
    random_search = RandomizedSearchCV(estimator=lrc,
                                       param_distributions=random_grid,
                                       n_iter=50,
                                       scoring='accuracy',
                                       cv=3,
                                       verbose=1,
                                       random_state=42,
                                       n_jobs=n_jobs)

    # Fit the random search model
    random_search.fit(x_train, y_train)

    print("The best hyperparameters from Random Search are:")
    print(random_search.best_params_)
    print("")
    print("The mean accuracy of a model with these hyperparameters is:")
    print(random_search.best_score_)

    best_lrc = random_search.best_estimator_
    print(best_lrc)
    return best_lrc


def get_SVM_random_grid(x_train, y_train):
    # C
    C = [.0001, .001, .01]

    # gamma
    gamma = [.0001, .001, .01, .1, 1, 10, 100]

    # degree
    degree = [1, 2, 3, 4, 5]

    # kernel
    kernel = ['linear', 'rbf', 'poly']

    # probability
    probability = [True, False]

    # Create the random grid
    random_grid = {'C': C,
                   'kernel': kernel,
                   'gamma': gamma,
                   'degree': degree,
                   'probability': probability
                   }

    pprint(random_grid)

    # First create the base model to tune
    svc = svm.SVC(random_state=42)

    # Definition of the random search
    random_search = RandomizedSearchCV(estimator=svc,
                                       param_distributions=random_grid,
                                       n_iter=50,
                                       scoring='accuracy',
                                       cv=3,
                                       verbose=1,
                                       random_state=42,
                                       n_jobs=n_jobs)

    # Fit the random search model
    random_search.fit(x_train, y_train)

    print("The best hyperparameters from Random Search are:")
    print(random_search.best_params_)
    print("")
    print("The mean accuracy of a model with these hyperparameters is:")
    print(random_search.best_score_)

    best_svm = random_search.best_estimator_
    print(best_svm)
    return best_svm


def get_random_forest_random_grid(x_train, y_train):
    # n_estimators
    n_estimators = [int(x) for x in np.linspace(start=200, stop=1200, num=5)]

    # max_features
    max_features = ['auto', 'sqrt']

    # max_depth
    max_depth = [int(x) for x in np.linspace(20, 120, num=5)]
    max_depth.append(None)

    # min_samples_split
    min_samples_split = [2, 5, 10]

    # min_samples_leaf
    min_samples_leaf = [1, 2, 4]

    # bootstrap
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    pprint(random_grid)

    rfc = RandomForestClassifier(random_state=42)

    # Definition of the random search
    random_search = RandomizedSearchCV(estimator=rfc,
                                       param_distributions=random_grid,
                                       n_iter=50,
                                       scoring='accuracy',
                                       cv=3,
                                       verbose=1,
                                       random_state=42,
                                       n_jobs=n_jobs)

    # Fit the random search model
    random_search.fit(x_train, y_train)

    print("The best hyperparameters from Random Search are:")
    print(random_search.best_params_)
    print("")
    print("The mean accuracy of a model with these hyperparameters is:")
    print(random_search.best_score_)

    best_rfc = random_search.best_estimator_
    print(best_rfc)
    return best_rfc


def get_gradient_boosting_random_grid(x_train, y_train):
    # n_estimators
    n_estimators = [200, 800]

    # max_features
    max_features = ['auto', 'sqrt']

    # max_depth
    max_depth = [10, 40]
    max_depth.append(None)

    # min_samples_split
    min_samples_split = [10, 30, 50]

    # min_samples_leaf
    min_samples_leaf = [1, 2, 4]

    # learning rate
    learning_rate = [0.1, 0.5]

    # subsample
    subsample = [0.5, 1.0]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'learning_rate': learning_rate,
                   'subsample': subsample}

    pprint(random_grid)

    # First create the base model to tune
    gbc = GradientBoostingClassifier(random_state=42)

    # Definition of the random search
    random_search = RandomizedSearchCV(estimator=gbc,
                                       param_distributions=random_grid,
                                       n_iter=50,
                                       scoring='accuracy',
                                       cv=3,
                                       verbose=1,
                                       random_state=42,
                                       n_jobs=n_jobs)

    # Fit the random search model
    random_search.fit(x_train, y_train)

    print("The best hyperparameters from Random Search are:")
    print(random_search.best_params_)
    print("")
    print("The mean accuracy of a model with these hyperparameters is:")
    print(random_search.best_score_)

    best_gbc = random_search.best_estimator_
    print(best_gbc)
    return best_gbc
