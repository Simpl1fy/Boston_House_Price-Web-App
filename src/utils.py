import os
import sys

import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from src.exception import CustomException
from sklearn.base import BaseEstimator, TransformerMixin

def save_object(file_path, obj):
    try:
        dirname = os.path.dirname(file_path)
        os.makedirs(dirname, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):

    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            # para = param[list(models.keys())[i]]

            # gs = GridSearchCV(model, para, cv=3)
            # gs.fit(X_train, y_train)

            # model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            y_test_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = y_test_score
        
        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


class ColumnDropper(BaseEstimator, TransformerMixin):
    def fit(self, X):
        return self
    
    def transform(self, X):
        return X.drop(['CRIM', 'ZN', 'CHAS', 'DIS', 'RAD', 'B', 'LSTAT'], axis=1)

class OutlierRemover(BaseEstimator, TransformerMixin):
    def fit(self, X):
        return self

    def transform(self, X):
        data = X[~(X['MEDV']>=50)]
        return data
