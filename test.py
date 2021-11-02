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
    df = pd.read_csv(path.join(PATH_DST, 'dataset_v3.csv')).drop(columns=['Unnamed: 0'])

    # mi calcolo le 

    teams = pd.read_csv(path.join(PATH_DST, 'dataset_v1_valid_country.csv')).drop(columns=['Unnamed: 0'])

    teams['games'] = 0
    teams['home_games'] = 0
    teams['away_games'] = 0
    teams['neutral_games'] = 0

    teams['wins'] = 0
    teams['home_wins'] = 0
    teams['away_wins'] = 0
    teams['neutral_wins'] = 0
    
    teams['home_defeats'] = 0
    teams['away_defeats'] = 0
    teams['neutral_defeats'] = 0

    teams['home_draws'] = 0
    teams['away_draws'] = 0
    teams['neutral_draws'] = 0

    teams['home_goals_scored'] = 0
    teams['away_goals_scored'] = 0
    teams['neutral_draws_scored'] = 0

    teams['home_goals_conceded'] = 0
    teams['away_goals_conceded'] = 0
    teams['neutral_goals_conceded'] = 0


    teams.to_csv(path.join(PATH_DST, 'dataset_v4.csv'))