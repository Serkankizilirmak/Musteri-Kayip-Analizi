import math
import os
import gc
import random
import pprint
import numpy as np  # For linear algebra
import pandas as pd  # For data manipulation

import warnings

warnings.filterwarnings("ignore")

from collections import Counter
from scipy import stats  # For statistics
from scipy.stats.contingency import association  # upgrade scipy to use this to calculate Cramer's V



"""scikit-learn modules"""
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import PowerTransformer  # convert to Gaussian-like data
from sklearn.feature_selection import chi2
from sklearn.metrics import matthews_corrcoef

import multiprocessing
import pickle, joblib

from IPython.display import Markdown, display


def print_unique_values(dataframe):
    for i in dataframe.columns:
        print(i)
        print(dataframe[i].unique())



left_padding = 21

SEED = 42

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(SEED)

def printmd(string):
    display(Markdown(string))

df = pd.read_csv("D:\SERKAN KIZILIRMAK\Python\AllProjects\Müşteri Kayıp Analizi (TelcoCustomer)\Data\TelcoCustomer(TR).csv")


def binning_feature(feature):
    plt.hist(df[feature])

    # set x/y labels and plot title
    plt.xlabel(f"{feature.title()}")
    plt.ylabel("Count")
    plt.title(f"{feature.title()} Bins")
    plt.show()

    bins = np.linspace(min(df[feature]), max(df[feature]), 4)
    printmd("**Value Range**")

    printmd(f"Low ({bins[0] : .2f} - {bins[1]: .2f})")
    printmd(f"Medium ({bins[1]: .2f} - {bins[2]: .2f})")
    printmd(f"High ({bins[2]: .2f} - {bins[3]: .2f})")
    group_names = ['Low', 'Medium', 'High']

    df.insert(df.shape[1] - 1, f'{feature}-binned',
                    pd.cut(df[feature], bins, labels=group_names, include_lowest=True))
    display(df[[feature, f'{feature}-binned']].head(10))

    # count values
    printmd("<br>**Binning Distribution**<br>")
    display(df[f'{feature}-binned'].value_counts())

    # plot the distribution of each bin
    plt.bar(group_names, df[f'{feature}-binned'].value_counts())
    # px.bar(data_canada, x='year', y='pop')

    # set x/y labels and plot title
    plt.xlabel(f"{feature.title()}")
    plt.ylabel("Count")
    plt.title(f"{feature.title()} Bins")
    plt.show()

