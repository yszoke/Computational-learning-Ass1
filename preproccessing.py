import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


def data_proccessing():

    raw_data = pd.read_excel('dataset.xlsx')

    # remove all peatures with threshold of 95%
    raw_data = raw_data.dropna(thresh=0.05 * len(raw_data), axis=1)
    features = list(raw_data)

    # remove features with those names
    to_remove = ['Influenza', 'Parainfluenza', 'Urine']
    for i in features:
        if any(x in i for x in to_remove):
            del raw_data[i]

    # remove features witch no in Table 1
    raw_data.drop(['Respiratory Syncytial Virus', 'CoronavirusNL63', 'Rhinovirus/Enterovirus', 'Coronavirus HKU1',
             'Chlamydophila pneumoniae', 'Adenovirus', 'Coronavirus229E', 'CoronavirusOC43', 'Inf A H1N1 2009',
             'Bordetella pertussis', 'Metapneumovirus', 'Strepto A'], axis=1, inplace=True)

    # remove features witch no in Table 1
    raw_data.drop(['Patient ID', 'Patient age quantile', 'Patient addmited to regular ward (1=yes, 0=no)',
                   'Patient addmited to semi-intensive unit (1=yes, 0=no)',
                   'Patient addmited to intensive care unit (1=yes, 0=no)'], axis=1, inplace=True)

    # remove all rows with threshold of 50%
    raw_data = raw_data.dropna(thresh=0.5 * raw_data.shape[1], axis=0)

    # print the Distribution in the class col
    print(raw_data['SARS-Cov-2 exam result'].value_counts())

    raw_data['SARS-Cov-2 exam result'] = np.where(raw_data['SARS-Cov-2 exam result'] == 'negative', 0, 1)

    print(raw_data)

    # fill numeric values with Iterative Imputer technique
    imp = IterativeImputer(max_iter=20)
    imp.fit(raw_data)
    imputed_df = imp.transform(raw_data)
    imputed_df = pd.DataFrame(imputed_df, columns=raw_data.columns)

    # check if there is still null
    print(imputed_df.isnull().values.any())

    print(imputed_df)

    return imputed_df


def build_model(data):
    X = data.loc[:, data.columns != 'SARS-Cov-2 exam result']
    y = data['SARS-Cov-2 exam result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)

    # Set a seed to ensure reproducibility
    seed = 7

    # Instantiate the Random Forest classifier
    rf = RandomForestClassifier(random_state=seed)

    # Number of rounds
    rounds = 20

    # Define the hyperparameter grid
    rf_param_grid = {'max_depth': [10, 50],
                     'n_estimators': [100, 200, 400]}

    # Create arrays to store the scores
    outer_scores = np.zeros(rounds)
    nested_scores = np.zeros(rounds)

    # Loop for each round
    for i in range(rounds):
        # Define both cross-validation objects (inner & outer)
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)

        # Non-nested parameter search and scoring
        clf = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=inner_cv)
        clf.fit(X_train, y_train)
        outer_scores[i] = clf.best_score_

        # Nested CV with parameter optimization
        nested_score = cross_val_score(clf, X=X_train, y=y_train, cv=outer_cv)
        nested_scores[i] = nested_score.mean()

    # Take the difference from the non-nested and nested scores
    score_difference = outer_scores - nested_scores

    print("Avg. difference of {:6f} with std. dev. of {:6f}."
          .format(score_difference.mean(), score_difference.std()))


def build_model2(data):
    X = data.loc[:, data.columns != 'SARS-Cov-2 exam result']
    y = data['SARS-Cov-2 exam result']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)

    seed = 7
    logreg = LogisticRegression(penalty='l1', solver='liblinear', multi_class='auto')
    LR_par = {'penalty': ['l1'], 'C': [0.5, 1, 5, 10], 'max_iter': [500, 1000, 5000]}

    rfc = RandomForestClassifier()
    param_grid = {'bootstrap': [True, False],
                  'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                  'max_features': ['auto', 'sqrt'],
                  'min_samples_leaf': [1, 2, 4, 25],
                  'min_samples_split': [2, 5, 10, 25],
                  'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

    mlp = MLPClassifier(random_state=seed)
    parameter_space = {'hidden_layer_sizes': [(10, 20), (10, 20, 10), (50,)],
                       'activation': ['tanh', 'relu'],
                       'solver': ['adam', 'sgd'],
                       'max_iter': [10000],
                       'alpha': [0.1, 0.01, 0.001],
                       'learning_rate': ['constant', 'adaptive']}

    gbm = GradientBoostingClassifier(min_samples_split=25, min_samples_leaf=25)
    param = {"loss": ["deviance"],
             "learning_rate": [0.15, 0.1, 0.05, 0.01, 0.005, 0.001],
             "min_samples_split": [2, 5, 10, 25],
             "min_samples_leaf": [1, 2, 4, 25],
             "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
             "max_features": ['auto', 'sqrt'],
             "criterion": ["friedman_mse"],
             "n_estimators": [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
             }

    svm = SVC(gamma="scale", probability=True)
    tuned_parameters = {'kernel': ('linear', 'rbf'), 'C': (1, 0.25, 0.5, 0.75)}

    def baseline_model(optimizer='adam', learn_rate=0.01):
        model = Sequential()
        model.add(Dense(100, input_dim=X.shape[1], activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(50, activation='relu'))  # 8 is the dim/ the number of hidden units (units are the kernel)
        model.add(Dense(4, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    inner_cv = KFold(n_splits=10, shuffle=True, random_state=seed)
    outer_cv = KFold(n_splits=10, shuffle=True, random_state=seed)

    models = []
    models.append(('LR', GridSearchCV(logreg, LR_par, cv=inner_cv, n_jobs=1)))
    models.append(('RFC', GridSearchCV(rfc, param_grid, cv=inner_cv, n_jobs=1)))
    models.append(('SVM', GridSearchCV(svm, tuned_parameters, cv=inner_cv, n_jobs=1)))
    models.append(('MLP', GridSearchCV(mlp, parameter_space, cv=inner_cv, n_jobs=1)))
    models.append(('GBM', GridSearchCV(gbm, param, cv=inner_cv, n_jobs=1)))


    results = []
    names = []
    scoring = 'accuracy'
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    for name, model in models:
        nested_cv_results = model_selection.cross_val_score(model, X, y, cv=outer_cv, scoring=scoring)
        results.append(nested_cv_results)
        names.append(name)
        msg = "Nested CV Accuracy %s: %f (+/- %f )" % (
        name, nested_cv_results.mean() * 100, nested_cv_results.std() * 100)
        print(msg)
        model.fit(X_train, Y_train)
        print('Test set accuracy: {:.2f}'.format(model.score(X_test, Y_test) * 100), '%')
        print("Best Parameters: \n{}\n".format(model.best_params_))
        print("Best CV Score: \n{}\n".format(model.best_score_))

if __name__ == "__main__":
    data = data_proccessing()
    # build_model(data)
    # build_model2(data)



