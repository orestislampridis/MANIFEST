import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.neural_network import MLPClassifier

n_jobs = -1


def get_logistic_regression_grid_search(x_train, y_train):
    # gender random search best hyper-parameters
    # {'solver': 'liblinear', 'penalty': 'l2', 'multi_class': 'ovr', 'C': 0.1}

    # fake news spreader phase A classifier random search best hyper-parameters
    # {'solver': 'liblinear', 'penalty': 'l2', 'multi_class': 'ovr', 'C': 0.1}

    # fake news spreader phase C classifier random search best hyper-parameters
    # {'solver': 'newton-cg', 'penalty': 'l2', 'multi_class': 'ovr', 'C': 0.1}

    # Create the parameter grid based on the results of random search
    C = [float(x) for x in np.linspace(start=0.1, stop=1, num=10)]
    multi_class = ['ovr']
    solver = ['newton-cg']
    penalty = ['l2']

    param_grid = {'C': C,
                  'multi_class': multi_class,
                  'solver': solver,
                  'penalty': penalty}

    # Create a base model
    lrc = LogisticRegression(random_state=42)

    # Manually create the splits in CV in order to be able to fix a random_state
    cv_sets = ShuffleSplit(n_splits=3, test_size=.33, random_state=42)

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=lrc,
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

    best_lrc = grid_search.best_estimator_
    print(best_lrc)
    return best_lrc


def get_SVM_grid_search(x_train, y_train):
    # gender random search best hyper-parameters
    # {'probability': False, 'kernel': 'poly', 'gamma': 10, 'degree': 2, 'C': 0.0001}

    # fake news spreader phase A classifier random search best hyper-parameters
    # {'probability': False, 'kernel': 'poly', 'gamma': 10, 'degree': 2, 'C': 0.0001}

    # fake news spreader phase C classifier random search best hyper-parameters
    # {'probability': False, 'kernel': 'poly', 'gamma': 10, 'degree': 2, 'C': 0.0001}

    # Create the parameter grid based on the results of random search
    C = [0.0001, 0.001, 0.01]
    degree = [1, 2, 3]
    gamma = [0.1, 1, 10]
    probability = [False]

    param_grid = [
        {'C': C, 'kernel': ['poly'], 'degree': degree, 'probability': probability},
        {'C': C, 'kernel': ['rbf'], 'gamma': gamma, 'probability': probability}
    ]

    # Create a base model
    svc = svm.SVC(random_state=42)

    # Manually create the splits in CV in order to be able to fix a random_state
    cv_sets = ShuffleSplit(n_splits=3, test_size=.33, random_state=42)

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=svc,
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

    best_svm = grid_search.best_estimator_
    print(best_svm)
    return best_svm


def get_random_forest_grid_search(x_train, y_train):
    # gender random search best hyper-parameters
    # {'n_estimators': 700, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 70,
    # 'bootstrap': False}

    # fake news spreader phase A classifier random search best hyper-parameters
    # {'n_estimators': 950, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'auto',
    # 'max_depth': 120, 'bootstrap': True}

    # fake news spreader phase C classifier random search best hyper-parameters
    # {'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt',
    # 'max_depth': 20, 'bootstrap': True}

    # Create the parameter grid based on the results of random search
    n_estimators = [200]
    min_samples_split = [8, 10, 12]
    min_samples_leaf = [1, 2, 4]
    max_features = ['sqrt']
    max_depth = [10, 20, 30]
    bootstrap = [True]

    param_grid = {
        'bootstrap': bootstrap,
        'max_depth': max_depth,
        'max_features': max_features,
        'min_samples_leaf': min_samples_leaf,
        'min_samples_split': min_samples_split,
        'n_estimators': n_estimators
    }

    # Create a base model
    rfc = RandomForestClassifier(random_state=8)

    # Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
    cv_sets = ShuffleSplit(n_splits=3, test_size=.33, random_state=8)

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=rfc,
                               param_grid=param_grid,
                               scoring='accuracy',
                               cv=cv_sets,
                               verbose=1,
                               n_jobs=n_jobs)

    # Fit the grid search to the data
    grid_search.fit(x_train, y_train)

    print("The best hyperparameters from the Grid Search are:")
    print(grid_search.best_params_)
    print("")
    print("The mean accuracy of a model with these hyperparameters is:")
    print(grid_search.best_score_)

    best_rfc = grid_search.best_estimator_
    print(best_rfc)
    return best_rfc


def get_gradient_boosting_grid_search(x_train, y_train):
    # gender random search best hyper-parameters
    # {'subsample': 1.0, 'n_estimators': 800, 'min_samples_split': 30, 'min_samples_leaf': 2, 'max_features': 'sqrt',
    # 'max_depth': None, 'learning_rate': 0.1}

    # fake news spreader phase A classifier random search best hyper-parameters
    # {'subsample': 0.5, 'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'auto',
    # 'max_depth': 40, 'learning_rate': 0.1}

    # fake news spreader phase C classifier random search best hyper-parameters
    # {'subsample': 0.5, 'n_estimators': 800, 'min_samples_split': 30, 'min_samples_leaf': 2, 'max_features': 'sqrt',
    # 'max_depth': 40, 'learning_rate': 0.1}

    # Create the parameter grid based on the results of random search
    max_depth = [20, 40, 60]
    max_features = ['sqrt']
    min_samples_leaf = [1, 2, 4]
    min_samples_split = [20, 30, 40]
    n_estimators = [800]
    learning_rate = [0.1]
    subsample = [0.5]

    param_grid = {
        'max_depth': max_depth,
        'max_features': max_features,
        'min_samples_leaf': min_samples_leaf,
        'min_samples_split': min_samples_split,
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'subsample': subsample
    }

    # Create a base model
    gbc = GradientBoostingClassifier(random_state=42)

    # Manually create the splits in CV in order to be able to fix a random_state
    cv_sets = ShuffleSplit(n_splits=3, test_size=.33, random_state=42)

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=gbc,
                               param_grid=param_grid,
                               scoring='accuracy',
                               cv=cv_sets,
                               verbose=1,
                               n_jobs=n_jobs)

    # Fit the grid search to the data
    grid_search.fit(x_train, y_train)
    best_gbc = grid_search.best_estimator_
    print(best_gbc)
    return best_gbc


def get_neural_network_grid_search(x_train, y_train):
    # fake news spreader phase C classifier random search best hyper-parameters
    # {'subsample': 0.5, 'n_estimators': 800, 'min_samples_split': 30, 'min_samples_leaf': 2, 'max_features': 'sqrt',
    # 'max_depth': 40, 'learning_rate': 0.1}

    # Create the parameter grid based on the results of random search
    hidden_layer_sizes = [100, 100, 100]
    activation = ['sqrt']
    solver = [1, 2, 4]
    alpha = [20, 30, 40]
    learning_rate = [800]

    param_grid = {
        'hidden_layer_sizes': [(100, 100, 100), (50, 50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }

    # Create a base model
    nn = MLPClassifier(max_iter=200)

    # Manually create the splits in CV in order to be able to fix a random_state
    cv_sets = ShuffleSplit(n_splits=3, test_size=.33, random_state=42)

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=nn,
                               param_grid=param_grid,
                               scoring='accuracy',
                               cv=cv_sets,
                               verbose=1,
                               n_jobs=n_jobs)

    # Fit the grid search to the data
    grid_search.fit(x_train, y_train)
    best_nn = grid_search.best_estimator_
    print(best_nn)
    return best_nn
