import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.logger import get_logger
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
#from sklearn.tree import DecisionTreeRegressor
from src.custom_Exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_fucntions import read_yaml, load_data
from scipy.stats import randint
import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:

    def __init__(self, train_path, test_path, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path =  model_output_path

        self.rf_param_dict = RNDOM_FOREST_PARAMS
        self.lr_param_dict = LINEAR_REGRESSION_PARAMS
        self.random_search_params =  RANDOM_SEARCH_PARAMS

    def load_and_split_data(self):
        try:
            logger.info(f"Loading data from {self.train_path}")
            train_df = load_data(self.train_path)

            logger.info(f"Loading data from {self.test_path}")
            test_df = load_data(self.test_path)

            X_train = train_df.drop(columns=["Target"])
            y_train = train_df["Target"]

            X_test = test_df.drop(columns=["Target"])
            y_test = test_df["Target"]

            logger.info("Data Splitted sucessfully for model trianing")

            return X_train, y_train, X_test, y_test

        except Exception as e:
            logger.error(f"Error while hanlding loading data {e}")
            raise CustomException("Failed to load data", e)
        
    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluation of model is started")

            y_pred = model.predict(X_test)

            r2_score = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred)

            return r2_score, mae, rmse
        except Exception as e:
            logger.error(f"Error while evaluating model '{model}' : {e}")
            raise CustomException("Failted to evaluate model", e)
        
    def train_rf(self, X_train , y_train, X_test, y_test):
        try:
            logger.info("Initializing Random forest model")

            rf_model = RandomForestRegressor(random_state=42)

            logger.info("Starting hyper parameters tuning for RandomForest")

            random_search = RandomizedSearchCV(
                estimator = rf_model,
                param_distributions= self.rf_param_dict,
                n_iter = self.random_search_params['n_iter'],
                cv = self.random_search_params['cv'],
                n_jobs=self.random_search_params['n_jobs'],
                verbose=self.random_search_params['verbose'],
                random_state=self.random_search_params['random_state'],
                scoring=self.random_search_params['scoring']

            )
            
            logger.info("Run Started from hyper parameter tuning")
            random_search.fit(X_train, y_train)
            logger.info("Run completed from hyper parameter tuining")

            best_params = random_search.best_params_
            best_rf_model = random_search.best_estimator_

            logger.info("best parameters are : {best_params}")
            logger.info("RF Testing on test data started")
            rf_r2, rf_mae, rf_rmse = evaluate_model(best_rf_model, X_test, y_test)
            logger.info("RF Testing on test data completed")
            return best_rf_model, rf_r2, rf_mae, rf_rmse
        
        except Exception as e:
            logger.error(f"Error while training RF model {e}")
            raise CustomException("Failed to train RF model", e)  
        


    def train_lr(self, X_train , y_train, X_test, y_test):
        try:
            logger.info("Initializing Random forest model")

            lr_model = LinearRegression()

            logger.info("Starting hyper parameters tuning for LR")

            lr_search = RandomizedSearchCV(
                estimator = lr_model,
                param_distributions= self.lr_param_dict,
                n_iter = self.random_search_params['n_iter'],
                cv = self.random_search_params['cv'],
                n_jobs=self.random_search_params['n_jobs'],
                verbose=self.random_search_params['verbose'],
                random_state=self.random_search_params['random_state'],
                scoring=self.random_search_params['scoring']

            )
            
            logger.info("Run Started from hyper parameter tuning for LR")
            lr_search.fit(X_train, y_train)
            logger.info("Run completed from hyper parameter tuining for LR")

            best_lr_params = lr_search.best_params_
            best_lr_model = lr_search.best_estimator_

            logger.info("best parameters are : {best_lr_params}")

            logger.info("RF Testing on test data started")
            lr_r2, lr_mae, lr_rmse = evaluate_model(best_lr_model, X_test, y_test)
            logger.info("RF Testing on test data completed")
            return best_lr_model, lr_r2, lr_mae, lr_rmse

        
        except Exception as e:
            logger.error(f"Error while training LR model {e}")
            raise CustomException("Failed to train LR model", e)
        
    

        
    def select_and_save(self, model):
        pass
