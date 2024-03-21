import os
import sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models

from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("initiating model trainer")
            X_train, X_test, y_train, y_test = (
                train_arr[:, :-1],
                test_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, -1]
            )
            logging.info("converted the data into training and testing array")

            models = {
                "Random Forest": RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                "CatBoost Regression": CatBoostRegressor(),
                "AdaBoost Regression": AdaBoostRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor()
            }

            logging.info("initialized models")

            # params = {
            #     "Decision Tree": {
            #         'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            #         # 'splitter':['best','random'],
            #         # 'max_features':['sqrt','log2'],
            #     },
            #     "Random Forest":{
            #         # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
            #         # 'max_features':['sqrt','log2',None],
            #         'n_estimators': [8,16,32,64,128,256]
            #     },
            #     "Gradient Boosting":{
            #         # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
            #         'learning_rate':[.1,.01,.05,.001],
            #         'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
            #         # 'criterion':['squared_error', 'friedman_mse'],
            #         # 'max_features':['auto','sqrt','log2'],
            #         'n_estimators': [8,16,32,64,128,256]
            #     },
            #     "Linear Regression":{},
            #     "XGBRegressor":{
            #         'learning_rate':[.1,.01,.05,.001],
            #         'n_estimators': [8,16,32,64,128,256]
            #     },
            #     "AdaBoost Regressor":{
            #         'learning_rate':[.1,.01,0.5,.001],
            #         # 'loss':['linear','square','exponential'],
            #         'n_estimators': [8,16,32,64,128,256]
            #     }
            # }

            # logging.info("initialized parameters")

            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test,
                                                y_test=y_test, models=models)
            
            logging.info("called the evaluate_models function")
            
            best_model_score = max(sorted(model_report.values()))
            logging.info("got the best model score")

            best_model_name  = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            logging.info("got the best model name")

            best_model = models[best_model_name]
            logging.info('created a best model object')

            if best_model_score < 0.6:
                raise CustomException("No Best Model Found")
            
            logging.info("Best model found for training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )
            logging.info('saved model pickle file')

            predicted = best_model.predict(X_test)
            logging.info('got a predicted value for x_test')

            score = r2_score(y_test, predicted)
            logging.info('returning r2_score')
            return (score, best_model_name)
        
        except Exception as e:
            raise CustomException(e, sys)
