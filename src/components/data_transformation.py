import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import pickle

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    

    def get_data_transformation_object():
        '''
        This function is responsible for returning the data transformation object
        '''

        try:
            input_column = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
            input_pipeline = Pipeline(steps=[
                ("min_max_scaler", MinMaxScaler())
            ])

            logging.info(f"Input columns is {input_column}")
            logging.info(f"Output Column is MEDV")

            preprocessor = ColumnTransformer(
                [
                    ("input_pipeline", input_pipeline, input_column)
                ]   
            )

            return (
                preprocessor
            )
        except Exception as e:
            raise CustomException(e, sys)
