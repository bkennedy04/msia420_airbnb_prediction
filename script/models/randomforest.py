import os
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier



class MyRandomForest():
    def __init__(self):
        self.SEED = 12345

    def data_reader(self):
        """
        read preprocessed train data

        Returns:
        data.frame
            preprocessed data.frame with lots of dummy variables
        """

        data_dir = os.path.join('../..', 'data')
        train_path = os.path.join(data_dir, 'train_binary_dummy.csv')
        train = pd.read_csv(train_path)

        return train

    def train_test(self, train):
        # Split data into response and predictors
        y = train['isNDF']
        x = train.drop('isNDF', axis=1)

        # Create training and test data tables
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=self.SEED)

        return x_train, x_test, y_train, y_test

    def fit_randomforest(self, x_train, y_train, max_features="auto"):

        num_trees = 100
        kfold = model_selection.KFold(n_splits=10, random_state=self.SEED)
        model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features, random_state=self.SEED)
        results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring='roc_auc')
        print(results.mean())

        model.fit(x_train, y_train)

        return model

    def parameter_tuning(self, model, X_train, y_train, param_grid):
        """
        Tune a tree based model using GridSearch, and return a model object with an updated parameters

        Parameters
        ----------
        model: sklearn's ensemble tree model
            the model we want to do the hyperparameter tuning.

        X_train: pandas DataFrame
            Preprocessed training data. Note that all the columns should be in numeric format.

        y_train: pandas Series

        param_grid: dict
            contains all the parameters that we want to tune for the responding model.


        Note
        ----------
        * we use kfold in GridSearchCV in order to make sure the CV Score is consistent with the score
          that we get from all the other function, including fit_bagging, fit_randomforest and fit_gbm.
        * We use model_selection.KFold with fixed seed in order to make sure GridSearchCV uses the same seed as model_selection.cross_val_score.

        """
        #     if 'n_estimators' in param_grid:
        #         model.set_params(warm_start=True)

        kfold = model_selection.KFold(n_splits=10, random_state=self.SEED)
        gs_model = GridSearchCV(model, param_grid, cv=kfold)
        gs_model.fit(X_train, y_train)

        # best hyperparameter setting
        print('best parameters:{}'.format(gs_model.best_params_))
        print('best score:{}'.format(gs_model.best_score_))

        # refit model on best parameters
        model.set_params(**gs_model.best_params_)
        model.fit(X_train, y_train)

        return model


if __name__=='__main__':

    # get preprocessed train data
    binary_rf = MyRandomForest()
    train = binary_rf.data_reader()

    # create train and test df
    x_train, x_test, y_train, y_test = binary_rf.train_test(train)

    # fit basic randomforest
    model_rf = binary_rf.fit_randomforest(x_train, y_train)

    # parameter tuning
    param_grid_rf_1 = {
        'max_depth': [None, 4, 6, 8, 10],
        'min_samples_leaf': [1, 3, 5, 7, 9],
        'max_features': ['auto', 'log2', None]
                      }
    model_rf2 = binary_rf.parameter_tuning(model_rf, x_train, y_train, param_grid_rf_1)
