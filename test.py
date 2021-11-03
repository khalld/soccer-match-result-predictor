import pandas as pd
import numpy as np
from libs.utils import fix_continent_matches, cumsum_graph, format_dataframe, get_continent_from_fifa, check_records_validity, convert_onehot, convert_onehot_simplified, add_weight, label_encode_df
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
plt.rcParams.update({'figure.figsize':(15,15), 'figure.dpi':100})

PATH_ORIGINAL_DST = 'dataset/original'
PATH_DST = 'dataset'

if __name__ == "__main__":

    # PARTE 1 ---- PREPARAZIONE DATASET

    df = pd.read_csv(path.join(PATH_DST, 'dataset_v3.csv')).drop(columns=['Unnamed: 0'])
    print("Preparing DataFrame")

    df['neutral'] = df['neutral'].astype(float)
    df['outcome'] = df['outcome'].replace({"Home": 1.0, "Draw": 0, "Away": 2})

    # elimino gli spazi dal dataset
    df['home_team'] = df['home_team'].str.replace(" ", "_")
    df['away_team'] = df['away_team'].str.replace(" ", "_")
    df['tournament'] = df['tournament'].str.replace(" ", "_")
    # non so se presente lo spazio ma modifico comunque
    df['country'] = df['country'].str.replace(" ", "_")
    df['city'] = df['city'].str.replace(" ", "_")

    print("Extracting label from categorical data")
    label_encode_df(df)

    print("CSV created correctly!")

    df.loc[:,'weight'] = df['tournament'].apply(add_weight)
    df.loc[:,'weight'] = 1 / ((2022 - df['year'].astype('int64'))*df['weight'])

    print("Added weight for each match")

    # prepara il modello a partire da qui
    df[['home_score','away_score']].apply(zscore)

    print("Applied z-score on home and away score")

    df.drop(columns=['year']).to_csv(path.join(PATH_DST, 'dataset_v4_TEMP.csv'))

    print("***"*5)
    print("ENDED COMPUTATION ON PART 1")
    print("***"*5)

    # PARTE 2 ---- CORRELAZIONE TRA VARIABILI

    # df = pd.read_csv(path.join(PATH_DST, 'dataset_v4_TEST_NOPUSH.csv'))

    # corr = df.corr(method='spearman').unstack().sort_values(ascending=False).drop_duplicates()

    # strong_corr = corr[(corr >= .7) & (corr <= 1)]
    # moderate_corr = corr[(corr >= .3) & (corr <= .7)]
    # weak_corr = corr[(corr >= .0) & (corr <= .3)]

    # print("Strong correlation: ")
    # print(strong_corr)
    # print("*=*"*5)
    # print("Moderate correlation: ")
    # print(moderate_corr)
    # print("*=*"*5)
    # print("Weak correlation: ")
    # print(weak_corr)

    # sns.heatmap(df.corr(),annot=True)
    # plt.show()