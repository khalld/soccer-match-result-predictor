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
plt.rcParams.update({'figure.figsize':(10,1), 'figure.dpi':100})

PATH_ORIGINAL_DST = 'dataset/original'
PATH_DST = 'dataset'

if __name__ == "__main__":

    # Parte 0 --- Aggiungo un peso al dataset

    # df = pd.read_csv(path.join(PATH_DST, 'dataset_v3.csv')).drop(columns=['Unnamed: 0'])
    # df['weight'] = 0
    # df.loc[:,'weight'] = df['tournament'].apply(add_weight)
    # df.loc[:,'weight'] = 1 / ( ( df['year'].max() + 1 - df['year'] ) * df['weight'] )
    # print("Added weight for each match. Some null value?")
    # print(df.isna().any(axis=None))
    # df.to_csv(path.join(PATH_DST, 'dataset_v3weight.csv'))

    # PARTE 2 --- Distribuzione di goal fatti/subiti di tute le quadre se segnano

    # df = pd.read_csv(path.join(PATH_DST, 'dataset_v3.csv')).drop(columns=['Unnamed: 0'])
    
    # sns.histplot(df[df['home_score']>0]['home_score'],kde=True,bins=30, color='g', label='Home Score')
    # sns.histplot(df[df['away_score']>0]['away_score'], kde=True, bins=30, color='r', label='Away Score')
    # plt.legend()
    # plt.xticks([i for i in range(1,21)])
    # plt.yticks([i for i in range(1000,13000,2000)])
    # plt.xlabel("Score")
    # plt.ylabel("Frequency")
    # plt.show()

    # PARTE 3 --- scatterplot tra home, away goals and outcome

    # sns.scatterplot(data=df, x="home_score", y="away_score", hue="outcome")#, style="time")
    # plt.show()

    # PARTE 4 --- preparazione dataset

    df = pd.read_csv(path.join(PATH_DST, 'dataset_v3weight.csv')).drop(columns=['Unnamed: 0'])
    print("Preparing DataFrame")

    df['home_score'] = zscore(df['home_score'])
    df['away_score'] = zscore(df['away_score'])
    df['neutral'] = df['neutral'].astype(float)
    df['outcome'] = df['outcome'].replace({"Home": 1.0, "Draw": 0, "Away": 2})

    # elimino gli eventuali spazi dal dataset
    df['home_team'] = df['home_team'].str.replace(" ", "_")
    df['away_team'] = df['away_team'].str.replace(" ", "_")
    df['tournament'] = df['tournament'].str.replace(" ", "_")
    df['country'] = df['country'].str.replace(" ", "_")
    df['city'] = df['city'].str.replace(" ", "_")

    print("Extracting label from categorical data")
    label_encode_df(df)
    print("CSVs created correctly!")
    print("Applied z-score on home and away score")

    df.drop(columns=['year']).to_csv(path.join(PATH_DST, 'dataset_v4_TEMP.csv'))

    print("***"*5 + "ENDED COMPUTATION first" + "***"*5)

    # PARTE 5 --- regplot to do ...


    # sns.regplot('home_score','outcome',df)



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

    # sns.regplot('al','window_glass',df)

    # sns.heatmap(df.corr(),annot=True)
    # plt.show()