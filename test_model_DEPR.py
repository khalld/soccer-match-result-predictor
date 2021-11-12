import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.formula.api import ols
from scipy.stats import poisson
from statsmodels.formula.api import logit
import statsmodels.formula.api as smf
import statsmodels.api as sm
from os import path
from libs.utils import format_columns, get_match_result

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
    # df = pd.read_csv(path.join(PATH_DST, 'dataset_v3.csv')).drop(columns=['Unnamed: 0'])
    # df = format_columns(df)
    # print("***"*5 + "STARTING LABEL ENCODING PROCESS" + "***"*5)
    # df = label_encoding(df)
    # df.to_csv(path.join(PATH_DST, 'dataset_v3_ENCODED.csv'))
    # print("***"*5 + "ENDED LABEL ENCODING PROCESS" + "***"*5)

    df = pd.read_csv(path.join(PATH_DST, 'dataset_v3_FORMATTED.csv')).drop(columns=['Unnamed: 0'])

    print("***"*5 + "STARTING MODEL" + "***"*5)
    matches_model_data = pd.concat([df[['home_team','away_team','home_score']].rename(columns={'home_team':'team', 'away_team':'opponent','home_score':'goals'}),
                df[['away_team','home_team','away_score']].rename(columns={'away_team':'team', 'home_team':'opponent','away_score':'goals'})])
    
    poisson_model = smf.glm(formula="goals ~ team + opponent", data=matches_model_data, family=sm.families.Poisson()).fit()
    print(poisson_model.summary())

    print("***"*5 + "PREDICTIONS:" + "***"*5)
    print(get_match_result(poisson_model, 'Germany', 'Spain'))
    print(get_match_result(poisson_model, 'Argentina', 'Germany'))
    print(get_match_result(poisson_model, 'England', 'Morocco'))
    print(get_match_result(poisson_model, 'Italy', 'Brazil'))

