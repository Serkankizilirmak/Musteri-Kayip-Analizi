import math
import os
import gc
import random
import pprint
import numpy as np  # For linear algebra
import pandas as pd  # For data manipulation

import warnings

warnings.filterwarnings("ignore")


"""scikit-learn modules"""
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import PowerTransformer  # convert to Gaussian-like data
from sklearn.feature_selection import chi2
from sklearn.metrics import matthews_corrcoef

import multiprocessing
import pickle, joblib




def print_unique_values(dataframe):
    for i in dataframe.columns:
        print(i)
        print(dataframe[i].unique())


################
left_padding = 21

SEED = 42

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(SEED)

##################

