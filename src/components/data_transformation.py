import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object 

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils import TargetLogger, InputLogger, ArrayToDf

import pickle

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    

    def get_data_transformation_object(self):
        '''
        This function is responsible for returning the data transformation object
        '''

        try:
            input_column = ['INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'LSTAT']
            target_column = ['MEDV']

            input_pipeline = Pipeline(steps=[
                ("min_max_scaler", MinMaxScaler()),
                ("Array to df", ArrayToDf()),
                ("log skew remove", InputLogger())
            ])
            

            logging.info(f"Input columns is {input_column}")
            logging.info(f"Output Column is {target_column}")

            tl = TargetLogger()


            return (
                input_pipeline,
                tl
            )
        except Exception as e:
            raise CustomException(e, sys)
    

    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("read train and test csv as dataframe")
            logging.info("obtaining preprocessing object")

            target_column_name = 'MEDV'

            preprocessing_obj, target_processor_obj = self.get_data_transformation_object()

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f'size of input_feature_train_df is {input_feature_train_df.shape}')

            logging.info("applying the preprocessor object to training and testing dataset")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            target_feature_train_df = target_processor_obj.fit_transform(target_feature_train_df)
            target_feature_test_df = target_processor_obj.transform(target_feature_test_df)


            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr
            )

        except Exception as e:
            raise CustomException(e, sys)
