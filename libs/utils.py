import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime
from os import path
from sklearn import preprocessing
from scipy.stats import poisson
import random

PATH_DST = 'dataset'

def get_outcome(home_score, away_score) -> str:
    if home_score > away_score:
        return 'Home'
    if away_score > home_score:
        return 'Away'
    if home_score == away_score:
        return 'Draw'

def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    df = df.astype({"home_score": float, "away_score": float})

    df['outcome'] = None
    # df['day_of_week'] = None
    # df['day'] = None
    # df['month'] = None
    df['year'] = None

    counter = 0

    for i in range(0, len(df)):
        counter = counter + 1
        print('*** Processing %d/%d ***' % (counter, len(df)), end='\r')

        # Uniformizzo nomi di squadre e continenti per standardizzarli al dataset FIFA utilizzato successivamente

        # --- Countries ----
        if df.loc[i]['country'] == 'United States':
            df.at[i, 'country'] = 'USA'
        elif df.loc[i]['country'] == 'Ivory Coast':
            df.at[i, 'country'] = "Côte d'Ivoire"
        elif df.loc[i]['country'] == 'Cape Verde':
            df.at[i, 'country'] = 'Cabo Verde'
        elif df.loc[i]['country'] == 'DR Congo':
            df.at[i, 'country'] = 'Congo DR'
        elif df.loc[i]['country'] == 'Iran':
            df.at[i, 'country'] = 'IR Iran'
        elif df.loc[i]['country'] == 'North Korea':
            df.at[i, 'country'] = 'Korea DPR'
        elif df.loc[i]['country'] == 'Saint Lucia':
            df.at[i, 'country'] = 'St. Lucia'
        elif df.loc[i]['country'] == 'Saint Vincent and the Grenadines':
            df.at[i, 'country'] = 'St. Vincent / Grenadines'
        elif df.loc[i]['country'] == 'South Korea':
            df.at[i, 'country'] = 'Korea Republic'
        elif df.loc[i]['country'] == 'United States Virgin Islands':
            df.at[i, 'country'] = 'US Virgin Islands'
        elif df.loc[i]['country'] == 'Saint Kitts and Nevis':
            df.at[i, 'country'] = 'St. Kitts and Nevis'

        # ---- Home teams ------
        if df.loc[i]['home_team'] == 'United States':
            df.at[i, 'home_team'] = 'USA'
        elif df.loc[i]['home_team'] == 'Ivory Coast':
            df.at[i, 'home_team'] = "Côte d'Ivoire"
        elif df.loc[i]['home_team'] == 'Cape Verde':
            df.at[i, 'home_team'] = 'Cabo Verde'
        elif df.loc[i]['home_team'] == 'DR Congo':
            df.at[i, 'home_team'] = 'Congo DR'
        elif df.loc[i]['home_team'] == 'Iran':
            df.at[i, 'home_team'] = 'IR Iran'
        elif df.loc[i]['home_team'] == 'North Korea':
            df.at[i, 'home_team'] = 'Korea DPR'
        elif df.loc[i]['home_team'] == 'Saint Lucia':
            df.at[i, 'home_team'] = 'St. Lucia'
        elif df.loc[i]['home_team'] == 'Saint Vincent and the Grenadines':
            df.at[i, 'home_team'] = 'St. Vincent / Grenadines'
        elif df.loc[i]['home_team'] == 'South Korea':
            df.at[i, 'home_team'] = 'Korea Republic'
        elif df.loc[i]['home_team'] == 'Saint Kitts and Nevis':
            df.at[i, 'home_team'] = 'St. Kitts and Nevis'
        elif df.loc[i]['home_team'] == 'United States Virgin Islands':
            df.at[i, 'home_team'] = 'US Virgin Islands'

        # ---- Away teams ------
        if df.loc[i]['away_team'] == 'United States':
            df.at[i, 'away_team'] = 'USA'
        elif df.loc[i]['away_team'] == 'Ivory Coast':
            df.at[i, 'away_team'] = "Côte d'Ivoire"
        elif df.loc[i]['away_team'] == 'Cape Verde':
            df.at[i, 'away_team'] = 'Cabo Verde'
        elif df.loc[i]['away_team'] == 'DR Congo':
            df.at[i, 'away_team'] = 'Congo DR'
        elif df.loc[i]['away_team'] == 'Iran':
            df.at[i, 'away_team'] = 'IR Iran'
        elif df.loc[i]['away_team'] == 'North Korea':
            df.at[i, 'away_team'] = 'Korea DPR'
        elif df.loc[i]['away_team'] == 'Saint Lucia':
            df.at[i, 'away_team'] = 'St. Lucia'
        elif df.loc[i]['away_team'] == 'Saint Vincent and the Grenadines':
            df.at[i, 'away_team'] = 'St. Vincent / Grenadines'
        elif df.loc[i]['country'] == 'Saint Kitts and Nevis':
            df.at[i, 'country'] = 'St. Kitts and Nevis'
        elif df.loc[i]['away_team'] == 'South Korea':
            df.at[i, 'away_team'] = 'Korea Republic'
        elif df.loc[i]['away_team'] == 'United States Virgin Islands':
            df.at[i, 'away_team'] = 'US Virgin Islands'

        # Estraggo i campi della colonna 'date'
        row_date = datetime.date.fromisoformat(df.iloc[i]['date'])
        # df.at[i, 'day'] = row_date.strftime('%d')
        # df.at[i, 'day_of_week'] = row_date.strftime('%A')
        # df.at[i, 'month'] = row_date.strftime('%B')
        df.at[i, 'year'] = int(row_date.strftime('%Y'))

        # Calcolo il risultato della partita
        df.at[i, 'outcome'] = get_outcome(df.iloc[i]['home_score'], df.iloc[i]['away_score'])


    # Calcolo il risultato della partita dopo i calci di rigore -- DEPRECATO
    # df_shootouts = pd.read_csv(path.join(PATH_ORIGINAL_DST, 'shootouts.csv')
    # if(df.iloc[i]['outcome'] == 'D'):
        # df.at[i, 'outcome'] = find_penalty(df.iloc[i], df_shootouts, df_shootouts.__len__() - 1)


    return df

def extract_continent_from_FIFA(df: pd.DataFrame, df_fifa: pd.DataFrame) -> pd.DataFrame:
    """ Ritorna un DataFrame con una colonna che indica il continente in cui è stato disputato il match """
    df['continent'] = ''

    # mi ricavo le confederazioni delle nazionali dal dataset FIFA
    df_fifa.drop(labels=['rank', 'rank_date', 'rank_change', 'total_points', 'previous_points', 'id'], axis=1, inplace=True)
    df_fifa.drop_duplicates(subset="country_full", keep="first", inplace=True)

    # eventuale dataframe con valori spuri
    df_nocontinent_found = pd.DataFrame(columns=df.columns)

    for i in range(0, len(df)):
        if df_fifa[(df_fifa.country_full == df.iloc[i]['country'])]['confederation'].values.__len__() == 1:
            confederation = df_fifa[(df_fifa.country_full == df.iloc[i]['country'])]['confederation'].values[0]
            
            if confederation in "CONCACAF":
                continent = "America"

            if confederation in "CONMEBOL":
                continent = "America"

            if confederation == "UEFA":
                continent = "Europe"

            if confederation == "AFC":
                continent = "Asia"
            
            if confederation == "CAF":
                continent = "Africa"

            if confederation == "OFC":
                continent = "Oceania"

            df.at[i, 'continent'] = continent
        else:
                current = df.iloc[i]['country']

                if (current == 'Bohemia'
                        or current == 'Soviet Union'
                        or current == 'Irish Free State'
                        or current == 'German DR'
                        or current == 'Saarland'
                        or current == 'Jersey'
                        or current == 'Northern Cyprus'
                        or current == 'Isle of Man'
                        or current == 'Guernsey'
                        or current == 'Bohemia and Moravia'
                        or current == 'Monaco' ):
                    df.at[i, 'continent'] = 'Europe'

                elif (current == 'British Guyana'
                        or current == 'Netherlands Guyana'
                        or current == 'French Guiana'
                        or current == 'Saint Kitts and Nevis'
                        or current == 'Éire'
                        or current == 'Guadeloupe'
                        or current == 'Martinique'
                        or current == 'Saint Martin'
                        or current == 'United States Virgin Islands'
                        or current == 'Sint Maarten'
                        or current == 'Greenland' ):
                    df.at[i, 'continent'] = 'America'

                elif (current == 'Manchuria'
                        or current == 'Ceylon'
                        or current == 'Burma'
                        or current == 'Malaya'
                        or current == 'Vietnam Republic'
                        or current == 'United Arab Republic'
                        or current == 'Vietnam DR'
                        or current == 'Taiwan'
                        or current == 'Kyrgyzstan'
                        or current == 'East Timor'
                        or current == 'Brunei'
                        or current == 'Yemen DPR'
                        or current == 'Yemen AR' ):
                    df.at[i, 'continent'] = 'Asia'

                elif (current == 'Northern Rhodesia'
                        or current == 'Tanganyika'
                        or current == 'French Somaliland'
                        or current == 'Belgian Congo'
                        or current == 'Southern Rhodesia'
                        or current == 'Réunion'
                        or current == 'Zanzibar' 
                        or current == 'Gold Coast'
                        or current == 'Nyasaland'
                        or current == 'Dahomey'
                        or current == 'Mali Federation'
                        or current == 'Upper Volta'
                        or current == 'Eswatini'
                        or current == 'Zaïre'
                        or current == 'Mayotte'
                        or current == 'Portuguese Guinea'
                        or current == 'Rhodesia' ):
                    df.at[i, 'continent'] = 'Africa'

                elif (df.iloc[i]['country'] == 'French Polynesia'
                        or current == 'New Hebrides'
                        or current == 'Lautoka'
                        or current == 'Northern Mariana Islands'
                        or current == 'Micronesia'
                        or current == 'Palau'
                        or current == 'Western Samoa' ):
                    df.at[i, 'continent'] = 'Oceania'

                else:
                    print("Exception at row: %d" % i)
                    df_nocontinent_found = df_nocontinent_found.append(df.iloc[i])

    if df_nocontinent_found.__len__() == 0:
        print("All rows are correctly updated")
    else:
        # per analizzare le eventuali righe spurie salvo il csv
        print(df_nocontinent_found)

    return df

def check_records_validity(df: pd.DataFrame, df_fifa: pd.DataFrame) -> pd.DataFrame:
    """ Aggiunge al dataframe la colonna is_valid che controlla la validità del match secondo le squadre """
    df_valid = df.copy()
    valid_country = df_fifa.country_full.drop_duplicates().sort_values().reset_index(drop=True)
    valid_country = valid_country.values
    print("Total country: %s"% (len(valid_country)))

    # assumo che la riga tutte le righe siano valide per diminuire il numero di iterazioni
    df_valid['is_valid'] = True
    
    for index, row in df_valid.iterrows():
        if (row.home_team not in valid_country and row.away_team not in valid_country):
            df_valid.at[index, 'is_valid'] = False

    return df_valid

def fix_continent_matches(all_years, df):
    """ Setto 0 gli anni in cui il continente non ha ospitato neanche una partita """
    not_played_years = set(all_years) - (set(all_years).intersection(df.year.values))

    for i in not_played_years:
        df = df.append({'year': i, 'matches': 0}, ignore_index=True)

    df = df.sort_values(by='year').reset_index().drop(columns=['index'])

    return df

def make_team_statistics(df: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    teams['games'] = 0
    teams['home_games'] = 0
    teams['away_games'] = 0
    teams['neutral_games'] = 0

    teams['wins'] = 0
    teams['home_wins'] = 0
    teams['away_wins'] = 0
    teams['neutral_wins'] = 0

    teams['defeats'] = 0
    teams['home_defeats'] = 0
    teams['away_defeats'] = 0
    teams['neutral_defeats'] = 0

    teams['draws'] = 0
    teams['home_draws'] = 0
    teams['away_draws'] = 0
    teams['neutral_draws'] = 0

    teams['goals_scored'] = 0
    teams['home_goals_scored'] = 0
    teams['away_goals_scored'] = 0
    teams['neutral_goals_scored'] = 0

    teams['goals_conceded'] = 0
    teams['home_goals_conceded'] = 0
    teams['away_goals_conceded'] = 0
    teams['neutral_goals_conceded'] = 0

    counter = 0
    for team in teams.team.values:
        print('*** Processing %d/%d ***' % (counter, len(teams.team.values)), end='\r')

        team_df = df.query("home_team == @team or away_team == @team")

        home_games_df = team_df.query("neutral == False and country == @team ")
        away_games_df = team_df.query("neutral == False and country != @team ")
        neutral_games_df = team_df.query("neutral == True ")

        teams.at[teams.team == team, 'games'] = len(team_df)
        teams.at[teams.team == team, 'home_games'] = len(home_games_df)
        teams.at[teams.team == team, 'away_games'] = len(away_games_df)
        teams.at[teams.team == team, 'neutral_games'] = len(neutral_games_df)

        query1 = "home_team == @team and outcome == 'Home' or away_team == @team and outcome == 'Away' "
        teams.at[teams.team == team, 'wins'] = len(team_df.query(query1))
        teams.at[teams.team == team, 'home_wins'] = len(home_games_df.query(query1))
        teams.at[teams.team == team, 'away_wins'] = len(away_games_df.query(query1))
        teams.at[teams.team == team, 'neutral_wins'] = len(neutral_games_df.query(query1))

        query2 = "home_team == @team and outcome == 'Away' or away_team == @team and outcome == 'Home' "
        teams.at[teams.team == team, 'defeats'] = len(team_df.query(query2))
        teams.at[teams.team == team, 'home_defeats'] = len(home_games_df.query(query2))
        teams.at[teams.team == team, 'away_defeats'] = len(away_games_df.query(query2))
        teams.at[teams.team == team, 'neutral_defeats'] = len(neutral_games_df.query(query2))

        query3 = "outcome == 'Draw' "
        teams.at[teams.team == team, 'draws'] = len(team_df.query(query3))
        teams.at[teams.team == team, 'home_draws'] = len(home_games_df.query(query3))
        teams.at[teams.team == team, 'away_draws'] = len(away_games_df.query(query3))
        teams.at[teams.team == team, 'neutral_draws'] = len(neutral_games_df.query(query3))

        teams.at[teams.team == team, 'goals_scored'] = team_df.query("home_team == @team")['home_score'].sum() + \
                                                       team_df.query("away_team == @team")['away_score'].sum()
        teams.at[teams.team == team, 'home_goals_scored'] = home_games_df.query("home_team == @team")[
                                                                'home_score'].sum() + \
                                                            home_games_df.query("away_team == @team")[
                                                                'away_score'].sum()
        teams.at[teams.team == team, 'away_goals_scored'] = away_games_df.query("home_team == @team")[
                                                                'home_score'].sum() + \
                                                            away_games_df.query("away_team == @team")[
                                                                'away_score'].sum()
        teams.at[teams.team == team, 'neutral_goals_scored'] = neutral_games_df.query("home_team == @team")[
                                                                   'home_score'].sum() + \
                                                               neutral_games_df.query("away_team == @team")[
                                                                   'away_score'].sum()

        teams.at[teams.team == team, 'goals_conceded'] = team_df.query("home_team == @team")['away_score'].sum() + \
                                                         team_df.query("away_team == @team")['home_score'].sum()
        teams.at[teams.team == team, 'home_goals_conceded'] = home_games_df.query("home_team == @team")[
                                                                  'away_score'].sum() + \
                                                              home_games_df.query("away_team == @team")[
                                                                  'home_score'].sum()
        teams.at[teams.team == team, 'away_goals_conceded'] = away_games_df.query("home_team == @team")[
                                                                  'away_score'].sum() + \
                                                              away_games_df.query("away_team == @team")[
                                                                  'home_score'].sum()
        teams.at[teams.team == team, 'neutral_goals_conceded'] = neutral_games_df.query("home_team == @team")[
                                                                     'away_score'].sum() + \
                                                                 neutral_games_df.query("away_team == @team")[
                                                                     'home_score'].sum()

        counter = counter + 1

    return teams


def show_cdf(df: pd.DataFrame, team_name: str) -> None:
    df = df.query("away_team == @team_name or home_team == @team_name")

    wins = df.query(
        "home_team == @team_name and outcome == 'Home' or away_team == @team_name and outcome == 'Away' ").value_counts(
        "year").sort_index(ascending=True).cumsum()
    draws = df.query("outcome == 'Draw'").value_counts("year").sort_index(ascending=True).cumsum()
    defeats = df.query(
        "home_team == @team_name and outcome == 'Away' or away_team == @team_name and outcome == 'Home' ").value_counts(
        "year").sort_index(ascending=True).cumsum()
    print("%d wins" % len(wins))
    print("%d draws" % len(draws))
    print("%d defeats" % len(defeats))
    
    plt.figure()
    plt.plot(wins)
    plt.plot(draws)
    plt.plot(defeats)
    plt.legend(['Wins', 'Draws', 'Defeats'])
    plt.xlabel("Years")
    plt.title(team_name)
    plt.show()

def show_cdf_splitted(df: pd.DataFrame, team_name: str) -> None:
    team_matches = df.query("away_team == @team_name or home_team == @team_name")

    home_matches = team_matches.query("neutral == False and country == @team_name")
    wins_home = home_matches.query("home_team == @team_name and outcome == 'Home' or away_team == @team_name and outcome == 'Away' ").value_counts("year").sort_index(ascending=True).cumsum()
    draws_home = home_matches.query("outcome == 'Draw'").value_counts("year").sort_index(ascending=True).cumsum()
    defeats_home = home_matches.query("home_team == @team_name and outcome == 'Away' or away_team == @team_name and outcome == 'Home' ").value_counts("year").sort_index(ascending=True).cumsum()
    
    away_matches = team_matches.query("neutral == False and country != @team_name")
    wins_away = away_matches.query("home_team == @team_name and outcome == 'Home' or away_team == @team_name and outcome == 'Away' ").value_counts("year").sort_index(ascending=True).cumsum()
    draws_away = away_matches.query("outcome == 'Draw'").value_counts("year").sort_index(ascending=True).cumsum()
    defeats_away = away_matches.query("home_team == @team_name and outcome == 'Away' or away_team == @team_name and outcome == 'Home' ").value_counts("year").sort_index(ascending=True).cumsum()

    neutral_matches = team_matches.query("neutral == True")
    wins_neutral = neutral_matches.query("home_team == @team_name and outcome == 'Home' or away_team == @team_name and outcome == 'Away' ").value_counts("year").sort_index(ascending=True).cumsum()
    draws_neutral = neutral_matches.query("outcome == 'Draw'").value_counts("year").sort_index(ascending=True).cumsum()
    defeats_neutral = neutral_matches.query("home_team == @team_name and outcome == 'Away' or away_team == @team_name and outcome == 'Home' ").value_counts("year").sort_index(ascending=True).cumsum()

    fig, axs = plt.subplots(1, 3)
    
    fig.suptitle("%s statistics" %team_name, fontsize="x-large")
    axs[0].set_title('Wins')
    axs[0].plot(wins_home, label='Home')
    axs[0].plot(wins_neutral, label='Neutral')
    axs[0].plot(wins_away, label="Away")
    
    axs[1].set_title('Draws')
    axs[1].plot(draws_home, label='Home')
    axs[1].plot(draws_neutral, label='Neutral')
    axs[1].plot(draws_away, label="Away")

    axs[2].set_title('Defeats')
    axs[2].plot(defeats_home, label='Home')
    axs[2].plot(defeats_neutral, label='Neutral')
    axs[2].plot(defeats_away, label="Away")

    for i in range(0,3):
        axs[i].set_xlabel("Years")
        axs[i].legend(loc='upper left')

    plt.show()

def do_label_encoding(df: pd.DataFrame) -> pd.DataFrame:
    label_encoder = preprocessing.LabelEncoder()

    # Estraggo label
    df_teams = pd.DataFrame()
    df_teams['name'] = df['home_team'].drop_duplicates().sort_values().reset_index().drop(labels=['index'], axis=1)
    df_teams['label'] = label_encoder.fit_transform(df_teams['name'])

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

    # Sostituisco nel dataframe i label
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

    # Salvo i csv dei label per usarli successivamente
    df_teams.to_csv(path.join("libs/csv" ,"coded_teams.csv"))
    df_tournament.to_csv(path.join("libs/csv" ,"coded_tournament.csv"))
    df_city.to_csv(path.join("libs/csv" ,"coded_city.csv"))
    df_country.to_csv(path.join("libs/csv" ,"coded_country.csv"))
    df_continent.to_csv(path.join("libs/csv" ,"coded_continent.csv"))

    return df

def format_columns(df: pd.DataFrame) -> pd.DataFrame:
    df['neutral'] = df['neutral'].astype(float)
    df['outcome'] = df['outcome'].replace({"Home": 1.0, "Draw": 0, "Away": 2})
    # elimino gli eventuali spazi vuoti dai nomi
    df['home_team'] = df['home_team'].str.replace(" ", "_")
    df['away_team'] = df['away_team'].str.replace(" ", "_")
    df['tournament'] = df['tournament'].str.replace(" ", "_")
    df['country'] = df['country'].str.replace(" ", "_")
    df['city'] = df['city'].str.replace(" ", "_")

    return df

def label_encoding(df: pd.DataFrame) -> pd.DataFrame:
    print("***"*5 + "DATAFRAME LOADED" +  "***"*5)

    print("Extracting label from categorical data..")
    df_encoded = do_label_encoding(df)
    print("Csv delle label salvati correttamente!")

    return df_encoded

def get_describe_values(df_input: pd.DataFrame, col_name: str):
    median = df_input[col_name].median()
    q1 = df_input[col_name].quantile(0.25) # 25th percentile / 1st quartile
    q3 = df_input[col_name].quantile(0.75) # 7th percentile / 3rd quartile
    iqr = q3-q1 # Range interquartile
    minimum  = q1-1.5*iqr # Valore minimo o |- marker nel box plot
    maximum = q3+1.5*iqr # Valore massimo o -| marker nel box plot
    return median, q1, q3, iqr, minimum, maximum

def remove_outliers(df_input: pd.DataFrame, col_name: str):
    # median, q1, q3, iqr, minimum, maximum = get_describe_values(df_input, col_name)
    # convenzione
    _, _, _, _, minimum, maximum = get_describe_values(df_input, col_name)
    df_output = df_input.loc[(df_input[col_name] > minimum) & (df_input[col_name] < maximum)]
    return df_output

# __DEPRECATO__?
def convert_onehot(home_team, away_team, tournament='Friendly', city='Rome', country='Italy', continent='Europe', neutral=0): # = 1 True = 0 False    
    df_teams = pd.read_csv(path.join("libs/csv" ,"coded_teams.csv"))
    df_tournament = pd.read_csv(path.join("libs/csv" ,"coded_tournament.csv"))
    df_city = pd.read_csv(path.join("libs/csv" ,"coded_city.csv"))
    df_country = pd.read_csv(path.join("libs/csv" ,"coded_country.csv"))
    df_continent = pd.read_csv(path.join("libs/csv" ,"coded_continent.csv"))

    predicted_home_team = df_teams.query("name == @home_team").label.values[0]
    predicted_away_team = df_teams.query("name == @away_team").label.values[0]
    predicted_tournament = df_tournament.query("name == @tournament").label.values[0]
    predicted_city = df_city.query("name == @city").label.values[0]
    predicted_country = df_country.query("name == @country").label.values[0]
    predicted_continent = df_continent.query("name == @continent").label.values[0]
    predicted_neutral = neutral

    return [[predicted_home_team, predicted_away_team, predicted_tournament, predicted_city, predicted_country, predicted_neutral, predicted_continent]]

# __DEPRECATO__?
def convert_onehot_simplified(home_team, away_team, neutral=True):
    df_teams = pd.read_csv(path.join("libs/csv" ,"coded_teams.csv"))

    predicted_home_team = df_teams.query("name == @home_team").label.values[0]
    predicted_away_team = df_teams.query("name == @away_team").label.values[0]
    if(neutral == True):
        predicted_neutral = 1
    else:
        predicted_neutral = 0

    return [[predicted_home_team, predicted_away_team, predicted_neutral]]


def get_proba_match(model, team1: str, team2: str, max_goals=10):
    # Media dei goal per ogni squadra
    t1_goals_avg = model.predict(pd.DataFrame(data={'team1': team1, 'team2': team2}, index=[1])).values[0]
    t2_goals_avg = model.predict(pd.DataFrame(data={'team1': team2, 'team2': team1}, index=[1])).values[0]
    
    # Probabilità di fare dei goal per ogni team
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [t1_goals_avg, t2_goals_avg]]
    
    # Prodotto dei due vettori per ottenere la matrice della partita
    match = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))
    
    # Probabilità per ogni possibile risultato
    t1_wins = np.sum(np.tril(match, -1))
    draw = np.sum(np.diag(match))
    t2_wins = np.sum(np.triu(match, 1))
    result_proba = [t1_wins, draw, t2_wins]
    
    # Adjust the proba to sum to one
    result_proba =  np.array(result_proba)/ np.array(result_proba).sum(axis=0,keepdims=1)
    team_pred[0] = np.array(team_pred[0])/np.array(team_pred[0]).sum(axis=0,keepdims=1)
    team_pred[1] = np.array(team_pred[1])/np.array(team_pred[1]).sum(axis=0,keepdims=1)
    
    return result_proba, [np.array(team_pred[0]), np.array(team_pred[1])]

# Sostituisci max_goals con gli outliers TODO
def get_match_result(model, team1: str, team2: str, max_draw=50, max_goals=10):
    # Get probabilità
    proba, score_proba = get_proba_match(model, team1, team2, max_goals)
    
    results = pd.Series([np.random.choice([team1, 'draw', team2], p=proba) for i in range(0,max_draw)]).value_counts()
    result = results.index[0]
    
    # Se il risultato non è un pareggio calcolo il numero goals da 1 a max_goals e 
    # e i goals del perdente da 0 ai goal del vincitore TODO puoi personalizzare
    if (result != 'draw'): 
        i_win, i_loose = (0,1) if result == team1 else (1,0)
        score_proba[i_win] = score_proba[i_win][1:]/score_proba[i_win][1:].sum(axis=0,keepdims=1)
        winner_score = pd.Series([np.random.choice(range(1, max_goals+1), p=score_proba[i_win]) for i in range(0,max_draw)]).value_counts().index[0]
        score_proba[i_loose] = score_proba[i_loose][:winner_score]/score_proba[i_loose][:winner_score].sum(axis=0,keepdims=1)
        looser_score = pd.Series([np.random.choice(range(0, winner_score), p=score_proba[i_loose]) for i in range(0,max_draw)]).value_counts().index[0]
        score = [winner_score, looser_score]

    # Se è un pareggio calcolo i goals e lo ripeto
    else:
        score = np.repeat(pd.Series([np.random.choice(range(0, max_goals+1), p=score_proba[0]) for i in range(0,max_draw)]).value_counts().index[0],2)
    looser = team2 if result == team1 else team1 if result != 'draw' else 'draw'

    return result, looser, score

def get_random_teams() -> str:
    country = pd.read_csv(path.join(PATH_DST, 'valid_country.csv')).drop(columns=['Unnamed: 0', 'confederation']).team.values
    team1 = random.choice(country)
    team2 = random.choice(country)
    if team1 == team2:
        team2 = random.choice(country)

    team1 = team1.replace(" ", "_")
    team2 = team2.replace(" ", "_")

    return team1, team2

def get_existent_matches(team1: str, team2:str)-> None:
    # dataframe di riferimento cablato
    df = pd.read_csv(path.join(PATH_DST, 'v3/dataset.csv')).drop(columns=['Unnamed: 0'])
    df = df.query("home_team == @team1 and away_team == @team2 or home_team == @team2 and away_team == @team1").drop(columns=['tournament','city','country','neutral','year','continent'])
    if len(df) == 0:
        print("Nessun precedente trovato")
    else:
        print("Trovati %d precedenti:" % len(df))
        draws = len(df.query("outcome == 'Draw'"))
        team1_wins = len(df.query("home_team == @team1 and outcome == 'Home' or away_team == @team1 and outcome == 'Away' "))
        team2_wins = len(df.query("home_team == @team2 and outcome == 'Home' or away_team == @team2 and outcome == 'Away' "))
        team1_defeats = len(df.query("home_team == @team1 and outcome == 'Away' or away_team == @team1 and outcome == 'Home' "))
        team2_defeats = len(df.query("home_team == @team2 and outcome == 'Away' or away_team == @team2 and outcome == 'Home' "))
        print("Vittorie di %s : %d" % (team1, team1_wins))
        print("Vittorie di %s : %d" % (team2, team2_wins))
        print("Pareggi: %d" % draws)
        print("Sconfitte di %s : %d" % (team1, team1_defeats))
        print("Sconfitte di %s : %d" % (team2, team2_defeats))

        team1_goals = pd.concat([df.query("home_team == @team1").home_score, df.query("away_team == @team1").away_score])
        team2_goals = pd.concat([df.query("home_team == @team2").home_score, df.query("away_team == @team2").away_score])

        print("Goal totali di %s: %d" % (team1, team1_goals.sum()))
        print("Goal totali di %s: %d" % (team2, team2_goals.sum()))

        # valore medio
        print("Mediano goal totali di %s: %d" % (team1, team1_goals.median()))
        print("Mediano goal totali di %s: %d" % (team2, team2_goals.median()))

        # valore mediano
        print("Media goal totali di %s: %d" % (team1, team1_goals.mean()))
        print("Media goal totali di %s: %d" % (team2, team2_goals.mean()))

def make_test(model, team_1_test: str = "", team_2_test: str = ""):
    proceed = False
    if(team_1_test == "" or team_2_test == ""):
        team_1_test, team_2_test = get_random_teams()
        proceed = True
    elif team_1_test == team_2_test:
        print("Parametri in input errati! Exit...")        
    
    print("Simulazione di %s vs %s" % (team_1_test, team_2_test))
    get_existent_matches(team_1_test, team_2_test)
    print("*"*5 + " Risultato simulazione " + "*"*5)
    print(get_match_result(model, team_1_test, team_2_test))

