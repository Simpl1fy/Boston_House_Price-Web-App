import sys
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts", "train.csv")
    test_data_path:str = os.path.join("artifacts", "test.csv")
    raw_data_path:str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    

    def initiate_data_ingestion(self):
        logging.info("Initiated data ingestion")
        try:
           column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
           df = pd.read_csv('notebook/data/housing.csv', header=None, names=column_names, delimiter=r"\s+")
           logging.info("Read the csv file with pandas as a dataframe")

           os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
           logging.info("Created a directory called artifact")

           logging.info("Saving the raw data as csv")
           df.to_csv(self.data_ingestion_config.raw_data_path, index=False)

           logging.info('initiated train test split')
           train_data, test_data = train_test_split(df, test_size=0.2)

           logging.info("saving the train and test data into csv")
           train_data.to_csv(self.data_ingestion_config.train_data_path, index=False)
           test_data.to_csv(self.data_ingestion_config.test_data_path, index=False)


           logging.info("data ingestion is completed")

           return(
               self.data_ingestion_config.train_data_path,
               self.data_ingestion_config.test_data_path
           )

        except Exception as e:
           raise CustomException(e, sys) 


# Running the code
if __name__ == "__main__":
    obj = DataIngestion()
    train_df,test_df = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_df, test_df)

    model_trainer = ModelTrainer()
    r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print(r2_score)