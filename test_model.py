import pandas as pd
import numpy as np
from libs.utils import fix_continent_matches, cumsum_graph, format_dataframe, get_continent_from_fifa, check_records_validity, convert_onehot, convert_onehot_simplified, add_weight, label_encoding
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
from statsmodels.formula.api import logit
from sklearn.tree import DecisionTreeClassifier 

from sklearn import tree
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
import pydotplus

import statsmodels.formula.api as smf
import statsmodels.api as sm

plt.style.use('ggplot')
plt.rcParams.update({'figure.figsize':(10,10), 'figure.dpi':100})

PATH_ORIGINAL_DST = 'dataset/original'
PATH_DST = 'dataset'


if __name__ == "__main__":

    # parto da questo dataset nel notebook
    # df = pd.read_csv(path.join(PATH_DST, 'dataset_v3.csv')).drop(columns=['Unnamed: 0'])
    # LABEL ENCODING
    # df = label_encoding(df)

    # df.drop(columns=['year']).to_csv(path.join(PATH_DST, 'dataset_v3_ENCODED.csv'))

    # print("***"*5 + "ENDED LABEL ENCODING PROCESS" + "***"*5)

    df = pd.read_csv(path.join(PATH_DST, 'dataset_v3_ENCODED.csv')).drop(columns=['Unnamed: 0'])
    # print(df.info())
    # print(df.corr())
    # print(df.columns)

    # print("***"*5 + " FIRST ITERATION " + "***"*5)
    # formula = "outcome ~ home_team + away_team + tournament + city + country + continent + neutral"
    # model = ols(formula=formula, data=df).fit()
    # print(model.summary())

    # il regressore non è significativo perché presenta un R-Squared basso.

    # i p-value di neutral e continent sono uguali a 0 quindi significa che sono statisticamente rilevanti

    # procedendo con la backward notiamo che il valore con p value più alto è tournament quindi lo elimino e procedo nuovamente

    # print("***"*5 + " SECOND ITERATION " + "***"*5)
    # formula = "outcome ~ home_team + away_team + city + country + continent + neutral"
    # model = ols(formula=formula, data=df).fit()
    # print(model.summary())

    # è aumentata la F-statistic ma non la R-squaded.
    # inoltre prob-f-statistic non è ancora nullo quindi non è statisticamente significativo
    # Continuo elemininando la variabile con p-value maggiore che è country

    # print("***"*5 + " THIRD ITERATION " + "***"*5)
    # formula = "outcome ~ home_team + away_team + city + continent + neutral"
    # model = ols(formula=formula, data=df).fit()
    # print(model.summary())


    # idem di sopra continuo eliminando city
    print("***"*5 + " FOURTH ITERATION " + "***"*5)
    formula = "outcome ~ home_team + away_team + continent + neutral"
    model = ols(formula=formula, data=df, ).fit()
    print(model.summary())

    # aggiungere un peso ad ogni riga del dal dataset non ha fatto migliorare per nulla l'algoritmo
    # neanche applicare lo z-score ai goals


    