import sys
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging


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
    obj.initiate_data_ingestion()