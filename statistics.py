from collections import Counter
from scipy import stats  # For statistics
from scipy.stats.contingency import association  # upgrade scipy to use this to calculate Cramer's V
import pandas as pd
import numpy as np
from IPython.display import Markdown, display
import math
import matplotlib.pyplot as plt

import plotly.graph_objs as go
import plotly.io as pio
##
def printmd(string):
    display(Markdown(string))
##

### Data Sets ####
df = pd.read_csv("D:\SERKAN KIZILIRMAK\Python\AllProjects\Müşteri Kayıp Analizi (TelcoCustomer)\Data\TelcoCustomer(TR)_binned.csv")

## Spearman ##

def cal_spearmanr(c1, c2):

    alpha = 0.05

    correlation, p_value = stats.spearmanr(df[c1], df[c2])

    print(f'{c1}, {c2} correlation : {correlation}, p : {p_value}')

    if p_value > alpha:
        print('Probably do not have monotonic relationship (fail to reject H0)')
    else:
        print('Probably have monotonic relationship (reject H0)')


#### Kendall Rank ####

def kendall_rank_correlation(feature1, feature2):

    coef, p_value = stats.kendalltau(df[feature1], df[feature2])
    print(f"Correlation between {feature1} and {feature2} ")
    print('Kendall correlation coefficient = %.5f, p = %.5f' % (coef, p_value))

    # interpret the significance
    alpha = 0.05
    if p_value > alpha:
        print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p_value)
    else:
        print('Samples are correlated (reject H0) p=%.3f' % p_value)
    print('----\n')

### Mann-Whitney U Test ###

def mannwhitneyu_correlation(feature1):
    stat, p_value = stats.mannwhitneyu(df[feature1], (df['Kayıp Durumu'] == 'Yes').astype(int))
    print(f"Correlation between {feature1} and Kayıp Durumu")
    print('Statistics = %.5f, p = %.5f' % (stat, p_value))

    # interpret the significance
    alpha = 0.05
    if p_value > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')
    print('----\n')

####  Polytomous(Nominal) with numeric ####

def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta

### Chi-Square ###

# alpha/significance = 0.05
# If p-value <= alpha: significant result, reject null hypothesis (H0), dependent
# If p-value > alpha: not significant result, fail to reject null hypothesis (H0), independent

def calculate_chi_square(feature1, feature2='Kayıp Durumu'):
    printmd(f"Correlation between **{feature1}** and **{feature2}**")
    crosstab = pd.crosstab(df[feature1], df[feature2])
    # display(crosstab)
    stat, p, dof, expected = stats.chi2_contingency(crosstab,correction=True)


    print(f'p-value : {p}, degree of freedom: {dof}')
    # print("expected frequencies :\n", expected)

    # interpret test-statistic
    prob = 0.95
    critical = stats.chi2.ppf(prob, dof)
    print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

    if abs(stat) >= critical:
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')

    # interpret p-value
    alpha = 1.0 - prob

    print('significance=%.3f, p=%.3f' % (alpha, p))
    if p <= alpha:
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')
    print('-----------------------------------\n')

# credit : https://machinelearningmastery.com/chi-squared-test-for-machine-learning

### Cramer's V ###
def cramers_v(x, y):
    """ calculate Cramers V statistic for categorial-categorial association.
      uses correction from Bergsma and Wicher,
      Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    confusion_matrix = pd.crosstab(x,y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

# credit : https://stackoverflow.com/a/46498792/11105356
###  Uncertainty Coefficient ###

def conditional_entropy(x,y):
  # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy

def theil_u(x,y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = stats.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x

### multivariate_analysis ###

def multivariate_analysis(cat_var_1, cat_var_2, cat_var_3, target_variable=df["Kayıp Durumu"]):
    fig, ax = plt.subplots(1, 1, figsize=(18, 5))
    font_size = 15
    cat_grouped_by_cat_target = pd.crosstab(index=[cat_var_1, cat_var_2, cat_var_3],
                                            columns=target_variable, normalize="index") * 100
    cat_grouped_by_cat_target.rename({"Yes": "% Churn", "No": "% Not Churn"}, axis=1, inplace=True)
    cat_grouped_by_cat_target.plot.bar(color=["green", "red"], ax=ax)
    ax.set_xlabel(f"{cat_var_1.name}, {cat_var_2.name}, {cat_var_3.name}", fontsize=font_size)
    ax.set_ylabel("Relative Frequency(%)", fontsize=font_size)
    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    plt.legend(loc="best")
    return plt.show()

