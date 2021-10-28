# from numba import jit # import decorator that allow to use gpu
from timeit import default_timer as timer
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def return_outcome(home_score, away_score):
    if home_score > away_score:
        return 'Home'
    if away_score > home_score:
        return 'Away'
    if home_score == away_score:
        return 'Draw'

def fix_continent_matches(all_years, df):
    # uniformizzo il range dei 3 continenti
    not_played_years = set(all_years) - (set(all_years).intersection(df.year.values))

    for i in not_played_years:
        df = df.append({'year': i, 'matches': 0}, ignore_index=True)

    df = df.sort_values(by='year').reset_index().drop(columns=['index'])

    return df

def cumsum_graph(df, team_name):
    df = df.query("away_team == @team_name or home_team == @team_name")

    wins = df.query(
        "home_team == @team_name and outcome == 'Home' or away_team == @team_name and outcome == 'Away' ").value_counts(
        "year").sort_index(ascending=True).cumsum()
    draws = df.query("outcome == 'Draw'").value_counts("year").sort_index(ascending=True).cumsum()
    losses = df.query(
        "home_team == @team_name and outcome == 'Away' or away_team == @team_name and outcome == 'Home' ").value_counts(
        "year").sort_index(ascending=True).cumsum()

    plt.figure()
    plt.plot(wins)
    plt.plot(draws)
    plt.plot(losses)
    plt.legend(['Wins', 'Draws', 'Losses'])
    plt.xlabel("Years")
    plt.title(team_name)
    plt.show()
