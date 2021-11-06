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

# codice utile riutilizzabile

    # Parte 0 --- Aggiungo un peso al dataset

    # df = pd.read_csv(path.join(PATH_DST, 'dataset_v3.csv')).drop(columns=['Unnamed: 0'])
    # df['weight'] = 0
    # df.loc[:,'weight'] = df['tournament'].apply(add_weight)
    # df.loc[:,'weight'] = 1 / ( ( df['year'].max() + 1 - df['year'] ) * df['weight'] )
    # print("Added weight for each match. Some null value?")
    # print(df.isna().any(axis=None))
    # df.to_csv(path.join(PATH_DST, 'dataset_v3weight.csv'))


    # PARTE 3 --- scatterplot tra home, away goals and outcome

    # sns.scatterplot(data=df, x="home_score", y="away_score", hue="outcome")#, style="time")
    # plt.show()


    # df['home_score'] = zscore(df['home_score'])
    # df['away_score'] = zscore(df['away_score'])
    # print("Applied z-score on home and away score")

if __name__ == "__main__":

    # parto da questo dataset nel notebook

    df = pd.read_csv(path.join(PATH_DST, 'dataset_v3.csv')).drop(columns=['Unnamed: 0'])

    # LABEL ENCODING

    # df = pd.read_csv(path.join(PATH_DST, 'dataset_v3weight.csv')).drop(columns=['Unnamed: 0'])

    print("***"*5 + "DATAFRAME LOADED" +  "***"*5)

    df['neutral'] = df['neutral'].astype(float)
    df['outcome'] = df['outcome'].replace({"Home": 1.0, "Draw": 0, "Away": 2})

    # elimino gli eventuali spazi dal dataset
    df['home_team'] = df['home_team'].str.replace(" ", "_")
    df['away_team'] = df['away_team'].str.replace(" ", "_")
    df['tournament'] = df['tournament'].str.replace(" ", "_")
    df['country'] = df['country'].str.replace(" ", "_")
    df['city'] = df['city'].str.replace(" ", "_")

    print("Extracting label from categorical data..")
    label_encode_df(df)
    print("Csv delle label salvati correttamente!")

    df.drop(columns=['year']).to_csv(path.join(PATH_DST, 'dataset_v3_ENCODED.csv'))

    print("***"*5 + "ENDED LABEL ENCODING PROCESS" + "***"*5)


    # df = pd.read_csv(path.join(PATH_DST, 'dataset_v4.csv')).drop(columns=['Unnamed: 0'])

    # formula = 'home_score ~ home_team + away_team + neutral'

    # datamodel = df.drop(columns=[ 'date', 'tournament', 'country', 'city', 'continent' ])
    # # model = ols(formula, datamodel).fit()

    # model = smf.glm(formula=formula, data=datamodel, family=sm.families.Poisson(), freq_weights=datamodel['weight'].values).fit()


    # print(model.summary())

    # print(model.predict({'home_team':1,'away_team': 207,'neutral': 1}))
    # print(model.predict({'home_team':22,'away_team': 1,'neutral': 1}))
    # print(model.predict({'home_team':25,'away_team': 122,'neutral': 1}))
    # print(model.predict({'home_team':27,'away_team': 207,'neutral': 1}))
    # print(model.predict({'home_team':27,'away_team': 207,'neutral': 0}))
    # print(model.predict({'home_team':27,'away_team': 59,'neutral': 0}))
    # print(model.predict({'home_team':207,'away_team': 17,'neutral': 0}))


    # sns.pairplot(df)
    # plt.show()

    # datamodel = df[['home_team', 'away_team', 'outcome']]
    # print(datamodel)

    # dt = DecisionTreeClassifier(max_depth=3)
    # dt.fit(datamodel.drop('outcome',axis=1),datamodel['outcome'])
    # print(dt.score(datamodel.drop('outcome',axis=1),datamodel['outcome']))

    # dt.predict([[159.0, 60.0]])
    # print( dt.predict([[ 1.0, 1.0 ]]) ) 

    # dotfile = StringIO()
    # export_graphviz(dt, out_file=dotfile)
    # graph = pydotplus.graph_from_dot_data(dotfile.getvalue())
    # graph.write_png("dtree.png")
    # Image(graph.create_png())
    # model = linear_model('outcome ~ home_team', df).fit()
    # print(model)

    # sns.regplot(x='home_score',y='outcome',data=df )
    # sns.regplot(x=df['home_score'], y=df['outcome'], logistic=True)
    # plt.show()



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