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

import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn.model_selection import train_test_split

plt.style.use('ggplot')
plt.rcParams.update({'figure.figsize':(10,10), 'figure.dpi':100})

PATH_ORIGINAL_DST = 'dataset/original'
PATH_DST = 'dataset'

def test_Linear_regr(df):
    print("===="*5 + " LINEAR REGRESSION " + "===="*5 )
    print("***"*5 + " FIRST ITERATION " + "***"*5)
    formula = "home_score ~ home_team + away_team + tournament + city + country + continent + neutral"
    model = ols(formula=formula, data=df).fit()
    print(model.summary())

    # il regressore non è significativo perché presenta un R-Squared basso.
    # i p-value di neutral e continent sono uguali a 0 quindi significa che sono statisticamente rilevanti
    # procedendo con la backward notiamo che il valore con p value più alto è tournament quindi lo elimino e procedo nuovamente

    print("***"*5 + " SECOND ITERATION " + "***"*5)
    formula = "home_score ~ home_team + away_team + city + country + continent + neutral"
    model = ols(formula=formula, data=df).fit()
    print(model.summary())

    # è aumentata la F-statistic ma non la R-squaded.
    # inoltre prob-f-statistic non è ancora nullo quindi non è statisticamente significativo
    # Continuo elemininando la variabile con p-value maggiore che è country

    print("***"*5 + " THIRD ITERATION " + "***"*5)
    formula = "home_score ~ home_team + away_team + city + continent + neutral"
    model = ols(formula=formula, data=df).fit()
    print(model.summary())

    # idem di sopra continuo eliminando city
    # -------- modello migliore con regressione lineare
    print("***"*5 + " FOURTH ITERATION " + "***"*5)
    formula = "home_score ~ home_team + away_team + continent + neutral"
    model = ols(formula=formula, data=df, ).fit()
    print(model.summary())

    return model

    # aggiungere un peso ad ogni riga del dal dataset non ha fatto migliorare per nulla l'algoritmo
    # neanche applicare lo z-score ai goals

def test_Poisson(df):
    print("===="*5 + " POISSON " + "===="*5 )

    # ------- inizi test per poisson
    print("***"*5 + " POISSON first ITERATION " + "***"*5)
    formula = "outcome ~ home_team + away_team + tournament + city + country + continent + neutral"
    model = smf.glm(formula=formula, data=df, family=sm.families.Poisson()).fit() # freq_weights, ricorda eventualmente di considerare..
    print(model.summary())

    # elimino tournament che è quello col p-value + alto
    print("***"*5 + " POISSON second ITERATION " + "***"*5)
    formula = "outcome ~ home_team + away_team + city + country + continent + neutral"
    model = smf.glm(formula=formula, data=df, family=sm.families.Poisson()).fit() # freq_weights, ricorda eventualmente di considerare..
    print(model.summary())

    # elimino country ...
    print("***"*5 + " POISSON third ITERATION " + "***"*5)
    formula = "outcome ~ home_team + away_team + city + continent + neutral"
    model = smf.glm(formula=formula, data=df, family=sm.families.Poisson()).fit() # freq_weights, ricorda eventualmente di considerare..
    print(model.summary())

    # elimino city
    print("***"*5 + " POISSON fourth ITERATION " + "***"*5)
    formula = "outcome ~ home_team + away_team + continent + neutral"
    model = smf.glm(formula=formula, data=df, family=sm.families.Poisson()).fit() # freq_weights, ricorda eventualmente di considerare..
    print(model.summary())

def test_Logit(df):
    #TODO

    print("***"*5 + " LOGIT FIRST ITERATION " + "***"*5)
    formula = "outcome ~ home_team + away_team + tournament + city + country + continent + neutral"
    model = logit(formula=formula, data=df).fit() # freq_weights, ricorda eventualmente di considerare..
    print(model.summary())

if __name__ == "__main__":
    np.random.seed(1234)

    print("________ PARTE 1 ________")
    # ------ PARTO DA QUESTA VERSIONE DEL DATASET DAL NOTEBOOK -------
    # df = pd.read_csv(path.join(PATH_DST, 'dataset_v4.csv')).drop(columns=['Unnamed: 0'])
    # print("***"*5 + "STARTING LABEL ENCODING PROCESS" + "***"*5)
    # df = label_encoding(df)
    # df.to_csv(path.join(PATH_DST, 'dataset_v4_ENCODED.csv'))
    # print("***"*5 + "ENDED LABEL ENCODING PROCESS" + "***"*5)

    print("________ PARTE 2 ________")
    # ------ DATASET CON LABEL ENCODING ----------
    df = pd.read_csv(path.join(PATH_DST, 'dataset_v4_ENCODED.csv')).drop(columns=['Unnamed: 0'])
    print(df.info())
    print(df.corr())
    print(df.columns)
    
    model = test_Linear_regr(df)
    # test_Poisson(df)


    # ------ TESTS -------

    df_train, df_test = train_test_split(df, test_size=0.25)


    # print(len(df_train))
    # print(len(df_test))
    
    # df_train, df_test = train_test_split(df, test_size=0.25)

    # print("***"*5 + " POISSON " + "***"*5)
    # formula1 = "outcome ~ home_team + away_team + home_score + away_score + tournament + city + country + continent + neutral"
    # formula2 = "outcome ~ home_team + away_team + home_score + away_score + continent + neutral"
    # formula3 = "outcome ~ home_team + away_team + home_score + away_score + neutral"

    # model = smf.glm(formula=formula1, data=df_train, family=sm.families.Poisson()).fit()
    # print(model.summary())
    # model = smf.glm(formula=formula2, data=df_train, family=sm.families.Poisson()).fit()
    # print(model.summary())
    # # model = smf.glm(formula=formula3, data=df_test, family=sm.families.Poisson()).fit()
    # # print(model.summary())
    poisson_predictions = model.get_prediction(df_test)
    print("PREDICTION")

    predicted_counts = poisson_predictions.summary_frame()

    # df.loc[df.Weight == "155", "Name"] = "John"

    print(predicted_counts)