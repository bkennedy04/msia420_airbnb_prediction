import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import model_selection


class MyLogisticRegression():
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

    def fit_model(self, x_train, y_train):

        # Fit logistic model
        logreg = LogisticRegression().fit(x_train, y_train)

        # Print out prediction accuracy for the data
        print('Model accuracy on train set: {:.2f}'.format(logreg.score(x_train, y_train)))

        coefficients = pd.DataFrame({"Feature": x_train.columns, "Coefficients": np.transpose(logreg.coef_[0])}).sort_values(by='Coefficients',
                                                                                                 ascending=False)

        return logreg, coefficients

    def get_cv_score(self, x_train, y_train, scoring):

        kfold = model_selection.KFold(n_splits=10, shuffle=False, random_state=self.SEED)
        modelCV = LogisticRegression()
        results = model_selection.cross_val_score(modelCV, x_train, y_train, cv=kfold, scoring=scoring)
        print("10-fold cross validation average %s: %.3f" % (scoring, results.mean()))

if __name__=='__main__':

    # get preprocessed train data
    binary_logreg = MyLogisticRegression()
    train = binary_logreg.data_reader()

    # create train and test df
    x_train, x_test, y_train, y_test = binary_logreg.train_test(train)

    # get cross-validation score
    binary_logreg.get_cv_score(x_train, y_train, scoring='roc_auc')
    binary_logreg.get_cv_score(x_train, y_train, scoring='accuracy')

    # fit final model
    logreg, coefficients = binary_logreg.fit_model(x_train, y_train)

