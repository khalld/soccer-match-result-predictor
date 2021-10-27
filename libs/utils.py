# from numba import jit # import decorator that allow to use gpu
from timeit import default_timer as timer
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def find_penalty(row, sht_csv: pd.DataFrame, sht_csv_len: int):
    for i in range(0, sht_csv_len):

        if (row['date'] == sht_csv.iloc[i]['date'] and row['home_team'] == sht_csv.iloc[i]['home_team'] and row[
            'away_team'] == sht_csv.iloc[i]['away_team']) is True:

            if (row['home_team'] == sht_csv.iloc[i]['winner']):
                return 'D-HP'

            if (row['away_team'] == sht_csv.iloc[i]['winner']):
                return 'D-AP'

    return 'D'


def return_outcome(home_score, away_score):
    if home_score > away_score:
        return 'Home'
    if away_score > home_score:
        return 'Away'
    if home_score == away_score:
        return 'Draw'


# @jit
def check_element(elem, array):
    if elem in array:
        return True

    return False


# @jit
def add_labels(elem, serie):
    if elem == serie['mean']:
        return 'mean'

    if elem == serie['25%']:
        return 'perc_25%'

    if elem == serie['50%']:
        return 'perc_50%'

    if elem == serie['75%']:
        return 'perc_75%'

def find_range(year):
    arr = ['1870-1879', '1880-1889', '1890-1899', '1900-1909', '1910-1919', '1920-1929', '1930-1939', '1940-1949',
           '1950-1959', '1960-1969', '1970-1979', '1980-1989', '1990-1999', '2000-2009', '2010-2020', '2021-2029']

    for i in arr:
        # print(i)
        x = int(i[:4])
        y = int(i[5:9]) + 1

        range_year = range(x, y)

        if year in range_year:
            return str(i)


# per fare il grafo devo utilizzare degli array con le stesse dimensioni
# anche per quegli anni in cui i contiennti non hanno giocato a calcio 
# ----- FIXA COME LO DICI -------
def fix_continent_matches(all_years, df):
    not_played_years = set(all_years) - (set(all_years).intersection(df.year.values))

    # df = pd.concat([pd.DataFrame([i, 0], columns=['year', 'matches']) for i in not_played_years], ignore_index=True)

    for i in not_played_years:
        df = df.append({'year': i, 'matches': 0}, ignore_index=True)

    df = df.sort_values(by='year').reset_index().drop(columns=['index'])

    return df


def extract_goals_per_year(years, dst):
    df = pd.DataFrame(data={
        'year': years,
        'n_matches': np.zeros(len(years), dtype=int),
        # home goals
        'hg': np.zeros(len(years), dtype=float),
        # away goals
        'ag': np.zeros(len(years), dtype=float),
        # total goals
        'tot': np.zeros(len(years), dtype=float)
    })

    for i in years:
        df.at[df['year'] == i, 'n_matches'] = dst[dst['year'] == i].home_score.__len__()
        df.at[df['year'] == i, 'hg'] = dst[dst['year'] == i].home_score.sum()
        df.at[df['year'] == i, 'ag'] = dst[dst['year'] == i].away_score.sum()
        df.at[df['year'] == i, 'tot'] = dst[dst['year'] == i].home_score.sum() + dst[dst['year'] == i].away_score.sum()

    return df


def cumsum_graph(df, team_name):
    df = df.query("away_team == @team_name or home_team == @team_name")

    wins = df.query("home_team == @team_name and outcome == 'Home' or away_team == @team_name and outcome == 'Away' ").value_counts("year").sort_index(ascending=True).cumsum()
    draws = df.query("outcome == 'Draw'").value_counts("year").sort_index(ascending=True).cumsum()
    losses = df.query("home_team == @team_name and outcome == 'Away' or away_team == @team_name and outcome == 'Home' ").value_counts("year").sort_index(ascending=True).cumsum()

    plt.figure()
    plt.plot(wins)
    plt.plot(draws)
    plt.plot(losses)
    plt.legend(['Wins', 'Draws', 'Losses'])
    plt.xlabel("Years")
    plt.title(team_name)
    plt.show()