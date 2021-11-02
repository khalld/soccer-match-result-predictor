import pandas as pd
import numpy as np
from libs.utils import fix_continent_matches, cumsum_graph, format_dataframe, get_continent_from_fifa, check_records_validity, convert_onehot, convert_onehot_simplified
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from os import path
from scipy.stats import spearmanr, pearsonr, kendalltau
import seaborn as sns
from statsmodels.formula.api import ols
from sklearn import preprocessing, linear_model
from scipy.stats import zscore
from scipy.stats import poisson

plt.style.use('ggplot')
plt.rcParams.update({'figure.figsize':(45,10), 'figure.dpi':100})

PATH_ORIGINAL_DST = 'dataset/original'
PATH_DST = 'dataset'

if __name__ == "__main__":
    df = pd.read_csv(path.join(PATH_DST, 'dataset_v2_continent.csv')).drop(columns=['Unnamed: 0'])
    # print(df.columns)

    valid_country = pd.read_csv(path.join(PATH_DST, 'dataset_v1_valid_country.csv')).drop(columns=['Unnamed: 0']).team.values

    df_country = df.country.drop_duplicates().values

    # print(df_country)

    not_valid_country = []

    for i in df_country:
        if i not in valid_country:
            not_valid_country.append(i)
        
    # print(not_valid_country)

    # trovati altri record non validi. Li elimino

    print("Lunghezza del dataframe prima del controllo: %s" % (len(df)) )

    for i, row in df.iterrows():
        if(row.country in not_valid_country):
            df.drop(i, inplace=True)

    print("Lunghezza del dataframe prima del controllo: %s" % (len(df)) )

    df = df.reset_index(drop=True).to_csv(path.join(PATH_DST, 'dataset_v3.csv'))
