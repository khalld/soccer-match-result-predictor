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
plt.rcParams.update({'figure.figsize':(15,15), 'figure.dpi':100})

PATH_ORIGINAL_DST = 'dataset/original'
PATH_DST = 'dataset'

def weight_from_tournament(value):
    df_tournaments = pd.read_csv(path.join("libs/csv" ,"coded_tournament.csv"))

    value = df_tournaments.query("label == @value").name.values[0]

    if 'FIFA' in value or 'UEFA' in value or 'CONCACAF' in value or 'AFC' in value:
        return 1
    else :
        return 100

if __name__ == "__main__":
    df = pd.read_csv(path.join(PATH_DST, 'dataset_v3.csv')).drop(columns=['Unnamed: 0', 'year'])

    df.loc[:,'weight'] = df['tournament'].apply(weight_from_tournament)
    df.loc[:,'weight'] = 1 / ((2022 - df['year'].astype('int64'))*df['weight'])

    # prepara il modello a partire da qui
    df[['home_score','away_score']].apply(zscore)

    # devo iniziare ad effettuare alcune modifiche al dataset per renderlo adatto al modello
    df['neutral'] = df['neutral'].astype(float)

    # outcome:
    #   1 home team win 
    #   0 draw
    #   2 away team win
    df['outcome'] = df['outcome'].replace({"Home": 1.0, "Draw": 0, "Away": 2})

    # elimino gli spazi dal dataset
    df['home_team'] = df['home_team'].str.replace(" ", "_")
    df['away_team'] = df['away_team'].str.replace(" ", "_")
    df['tournament'] = df['tournament'].str.replace(" ", "_")
    # non so se presente lo spazio ma modifico comunque
    df['country'] = df['country'].str.replace(" ", "_")
    df['city'] = df['city'].str.replace(" ", "_")

    # print("only to be sure... %d %d " % len(df['home_team'].drop_duplicates()), len(df['away_team'].drop_duplicates()))
    # dovresti inserire una funzioncina per verificare che gli array siano esattamente uguali

    label_encoder = preprocessing.LabelEncoder()

    # Prepare DF to label encoding for home_team and away_team
    df_teams = pd.DataFrame()
    df_teams['name'] = df['home_team'].drop_duplicates().sort_values().reset_index().drop(labels=['index'], axis=1)
    df_teams['label'] = label_encoder.fit_transform(df_teams['name'])

    # Tournaments label encoding
    df_tournament = pd.DataFrame()
    df_tournament['name'] = df['tournament'].drop_duplicates().sort_values().reset_index().drop(labels=['index'], axis=1)
    df_tournament['label'] = label_encoder.fit_transform(df_tournament['name'])

    df_country = pd.DataFrame()
    df_country['name'] = df['country'].drop_duplicates().sort_values().reset_index().drop(labels=['index'], axis=1)
    df_country['label'] = label_encoder.fit_transform(df_country['name'])

    df_city = pd.DataFrame()
    df_city['name'] = df['city'].drop_duplicates().sort_values().reset_index().drop(labels=['index'], axis=1)
    df_city['label'] = label_encoder.fit_transform(df_city['name'])

    df_continent = pd.DataFrame()
    df_continent['name'] = df['continent'].drop_duplicates().sort_values().reset_index().drop(labels=['index'], axis=1)
    df_continent['label'] = label_encoder.fit_transform(df_continent['name'])

    for i, row in df.iterrows():

        df.at[i, 'home_team'] = df_teams.query("name == @row.home_team")["label"].values.astype(float)[0]
        df.at[i, 'away_team'] = df_teams.query("name == @row.away_team")["label"].values.astype(float)[0]
        df.at[i, 'tournament'] = df_tournament.query("name == @row.tournament")["label"].values.astype(float)[0]
        df.at[i, 'country'] = df_country.query("name == @row.country")["label"].values.astype(float)[0]
        df.at[i, 'city'] = df_city.query("name == @row.city")["label"].values.astype(float)[0]
        df.at[i, 'continent'] = df_continent.query("name == @row.continent")["label"].values.astype(float)[0]

    df['home_team'] = df['home_team'].astype(float)
    df['away_team'] = df['away_team'].astype(float)
    df['tournament'] = df['tournament'].astype(float)
    df['country'] = df['country'].astype(float)
    df['city'] = df['city'].astype(float)
    df['continent'] = df['continent'].astype(float)

    # df_teams.to_csv(path.join("libs/csv" ,"coded_teams.csv"))
    # df_tournament.to_csv(path.join("libs/csv" ,"coded_tournament.csv"))
    # df_city.to_csv(path.join("libs/csv" ,"coded_city.csv"))
    # df_country.to_csv(path.join("libs/csv" ,"coded_country.csv"))
    # df_continent.to_csv(path.join("libs/csv" ,"coded_continent.csv"))


    corr = df.corr(method='spearman').unstack().sort_values(ascending=False).drop_duplicates()

    strong_corr = corr[(corr >= .7) & (corr <= 1)]
    moderate_corr = corr[(corr >= .3) & (corr <= .7)]
    weak_corr = corr[(corr >= .0) & (corr <= .3)]

    print("Strong correlation: ")
    print(strong_corr)
    print("*=*"*5)
    print("Moderate correlation: ")
    print(moderate_corr)
    print("*=*"*5)
    print("Weak correlation: ")
    print(weak_corr)

    sns.heatmap(df.corr(),annot=True)
    plt.show()