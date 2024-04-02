import os
import sys
import numpy as np
import pandas as pd

import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from src.exception import CustomException
from sklearn.base import BaseEstimator, TransformerMixin

class TargetLogger(BaseEstimator, TransformerMixin):
    def fit(self,X):
        return self
    def transform(self, X):
        y = np.log1p(X)
        return y

class InputLogger(BaseEstimator, TransformerMixin):
    def fit(self, X):
        return self
    def transform(self, X):
        for col in X.columns:
            if np.abs(X[col].skew()) > 0.3:
                X[col] = np.log1p(X[col])
        return X

class ArrayToDf(BaseEstimator, TransformerMixin):
    def fit(self, X):
        return self
    def transform(self, X):
        column_names = ['INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'LSTAT']
        new_df = pd.DataFrame(data=X, columns=column_names)
        return new_df


def save_object(file_path, obj):
    try:
        dirname = os.path.dirname(file_path)
        os.makedirs(dirname, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train,X_test,y_test,models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            # para=param[list(models.keys())[i]]

            # gs = GridSearchCV(model,para,cv=3)
            # gs.fit(X_train,y_train)

            # model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

