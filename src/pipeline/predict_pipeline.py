import sys
import pandas as pd
import numpy as np
import os

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            print("Before Loading")
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                indus,
                nox,
                rm,
                age,
                dis,
                tax,
                ptratio,
                lstat):
        self.indus = indus
        self.nox = nox
        self.rm = rm
        self.age = age
        self.dis = dis
        self.tax = tax
        self.ptratio = ptratio
        self.lstat = lstat

    def get_data_as_dataframe(self):
        try:
            custom_data_dict = {
                    "INDUS" : [self.indus],
                    "NOX" : [self.nox],
                    "RM" : [self.rm],
                    "AGE": [self.age],
                    "DIS" : [self.dis],
                    "TAX" : [self.tax],
                    "PTRATIO" : [self.ptratio],
                    "LSTAT" : [self.lstat]
            } 

            return pd.DataFrame(custom_data_dict)
        except Exception as e:
            raise CustomException(e, sys)
