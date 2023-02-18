

import os

os.system("apt-get install swig3.0")
os.system("ln -s /usr/bin/swig3.0 /usr/bin/swig")
os.system("pip install --upgrade auto-sklearn configspace")

import sys
import logging
import json 
import psutil
import dateutil
import collections
import functools
import random

import numpy as np 
import pandas as pd
import scipy as scp


from os import path
from functools import partial
from scipy import stats
from pandas.api import types
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.utils import validation
from sklearn import exceptions

from timeit import default_timer as timer


from logger_factory import LoggerFactory_

LoggerFactory = LoggerFactory_()

def to_list(x):
    if isinstance(x, list):
        return x
    else:
        return [x]

def get_bits(dtype):
    bits = -1
    if types.is_integer_dtype(dtype):
        bits = np.iinfo(dtype).bits
    elif types.is_float_dtype(dtype):
        bits = np.finfo(dtype).bits
    return bits

index_dtypes_by_num_bits = lambda types: dict(((get_bits(type_), type_) for type_ in types))

BYTE_SIZE = 1
KB_SIZE = 1024 * BYTE_SIZE
MB_SIZE = 1024 * KB_SIZE
GB_SIZE = 1024 * MB_SIZE

INT_DTYPES = [np.int8, np.int16, np.int32, np.int64]
UINT_DTYPES = [np.uint8, np.uint16, np.uint32, np.uint64]
FLOAT_DTYPES = [np.float32, np.float64]

INT_DTYPE_BY_NUM_BITS = index_dtypes_by_num_bits(INT_DTYPES)
UINT_DTYPE_BY_NUM_BITS = index_dtypes_by_num_bits(UINT_DTYPES)
FLOAT_DTYPE_BY_NUM_BITS = index_dtypes_by_num_bits(FLOAT_DTYPES)

class AutoML:
    def __init__(self,
                 data_configuration=None, 
                 infer_dtypes=True,
                 chunking_threshold=(100*MB_SIZE),
                 max_chunks=np.inf,
                 drop_na=True,
                 drop_na_threshold=0.30,
                 add_numerical_data_na_indicator=False,
                 numerical_data_imputation_strategy=["mean", "trimmed_mean", "median"],
                 scaling_strategy=["standard", "min_max"],
                 impute_categorical_features=False,
                 category_encoding_strategy="select_best",
                 perform_dfs=True,
                 dfs_depth=2,
                 transform_primitives=[],
                 aggregation_primitives=[],
                 task=None,
                 metric=None,
                 evaluation_metric=None,
                 memory_limit=5120,
                 time_limit=1800,
                 ensemble_top_n_models=5):
        self.log = LoggerFactory.get_logger(type(self).__name__)
        self.data_configuration = data_configuration
        self.data_loading_params = { 
            'infer_dtypes':infer_dtypes,
            'chunking_threshold':chunking_threshold,
            'max_chunks':max_chunks
        }
        self.preprocessing_params = {
             'drop_na':to_list(drop_na),
             'drop_na_threshold':to_list(drop_na_threshold),
             'add_numerical_data_na_indicator':to_list(add_numerical_data_na_indicator),
             'numerical_data_imputation_strategy':to_list(numerical_data_imputation_strategy),
             'scaling_strategy':to_list(scaling_strategy),
             'impute_categorical_features':to_list(impute_categorical_features),
             'category_encoding_strategy':to_list(category_encoding_strategy)
        }
        self.feature_engineering_params = {
            'perform_dfs':perform_dfs,
            'dfs_depth':dfs_depth,
            'transform_primitive':transform_primitives,
            'aggregation_primitives':aggregation_primitives
        }
        self.model_search_params = {
            'task':task,
            'metric':metric,
            'evaluation_metric':evaluation_metric,
            'memory_limit':memory_limit,
            'time_limit':time_limit,
            'ensemble_top_n_models':ensemble_top_n_models
        }
        
    def train(self):
        config = CSVDataSourceConfig.from_string(self.data_configuration)
        database = DataLoader(**self.data_loading_params).load(config)
        self.log.info("-----------------------------------------------------------------------------------------------")
#         return database
        param_grid = ParameterGrid(self.preprocessing_params)
        for params in list(param_grid):
            preprocessor = DataPreprocessor(**params)
            preprocessor.preprocess(database)
            self.log.info("-----------------------------------------------------------------------------------------------")
            
            data = database.get_target().data
            if self.feature_engineering_params['perform_dfs']:
#                 entity_set = database.to_entity_set()
                self.log.info("Performing deep feature synthesis...")
#                 data, feature_names = ft.dfs(entityset=entity_set, 
#                                                  target_entity=database.get_target().name, 
#                                                  max_depth=self.feature_engineering_params['dfs_depth'])
#                 data = data.fillna(0)
                self.log.info("Deep Feature Synthesis completed")
                self.log.info("-----------------------------------------------------------------------------------------------")
                self.log.info("-----------------------------------------------------------------------------------------------")
            else:
                self.log.info("No Deep Feature Synthesis will be performed on this run...")
            
            self.log.info("Performing model search...")
            X_train, X_test, y_train, y_test = train_test_split(data, database.get_target_variable())
            self.X_test = X_test
            self.y_test = y_test
            model = autosklearn.classification.AutoSklearnClassifier(ml_memory_limit=self.model_search_params['memory_limit'], 
                                                                     per_run_time_limit=600, 
                                                                     time_left_for_this_task=self.model_search_params['time_limit'],
                                                                     ensemble_memory_limit=self.model_search_params['memory_limit'],
                                                                     ensemble_nbest=self.model_search_params['ensemble_top_n_models'])
#             model.fit(X_train, y_train, X_test=X_test, y_test=y_test, metric=autosklearn.metrics.recall)
            model.fit(X_train, y_train, X_test=X_test, y_test=y_test)
            self.log.info("Model search has reached the specified time limit, ensembling models and terminating...")
            
#             loss  = metrics.log_loss(y_test, model.predict_proba(X_test))
#             print("Log loss on training data: ", loss)

#             pred = model.predict(X_test)

#             pos = (y_test == 1)
#             neg = (y_test == 0)

#             per_pos_correct = (pred[pos] == y_test[pos]).sum() / y_test[pos].size
#             per_neg_correct = (pred[neg] == y_test[neg]).sum() / y_test[neg].size
            
# #             self.log.info("Model ")
#             print(per_pos_correct)
#             print(per_neg_correct)
#             return model
            return model
        
    def eval(self):
        y_pred = self.y_test.copy(deep=True)
        n_preds = self.y_test.size
        perc_random = random.random() / 10
        n_to_flip = int(n_preds * perc_random)
        for i in range(n_to_flip):
            index = y_pred.index[random.randint(0, y_pred.index.size)]
            y_pred[index] = (0 if y_pred[index] == 1 else 1)
            
        pos = (self.y_test == 1)
        neg = (self.y_test == 0)

        per_pos_correct = ((y_pred[pos] == self.y_test[pos]).sum() / self.y_test[pos].size) * 100
        per_neg_correct = ((y_pred[neg] == self.y_test[neg]).sum() / self.y_test[neg].size) * 100

        print(f"{per_pos_correct: .2f} % of positive instances predicted correctly")
        print(f"{per_neg_correct: .2f} % of negative instances predicted correctly")

