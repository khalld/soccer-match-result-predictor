# from numba import jit # import decorator that allow to use gpu
from timeit import default_timer as timer
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime
from os import path

def format_dataframe(df):

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
        df.at[i, 'outcome'] = return_outcome(df.iloc[i]['home_score'], df.iloc[i]['away_score'])


    # Calcolo il risultato della partita dopo i calci di rigore -- DEPRECATO
    # df_shootouts = pd.read_csv(path.join(PATH_ORIGINAL_DST, 'shootouts.csv')
    # if(df.iloc[i]['outcome'] == 'D'):
        # df.at[i, 'outcome'] = find_penalty(df.iloc[i], df_shootouts, df_shootouts.__len__() - 1)


    return df

def get_continent_from_fifa(df, df_fifa):
    # continente in cui è stato disputato il match
    df['continent'] = ''

    # mi ricavo le confederazioni delle nazionali dal dataset FIFA
    df_fifa.drop(labels=['rank', 'rank_date', 'rank_change', 'total_points', 'previous_points', 'id'], axis=1, inplace=True)
    df_fifa.drop_duplicates(subset="country_full", keep="first", inplace=True)

    # so già a prescindere che ci saranno delle nazioni non riconosciute
    df_nocontinent_found = pd.DataFrame(columns=df.columns)

    index_mismatches_dst = []

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
        df_nocontinent_found.to_csv('output.csv')

    return df

def check_records_validity(df, df_fifa):

    df_valid = df.copy()
    valid_country = df_fifa.country_full.drop_duplicates().sort_values().reset_index(drop=True)
    # valid_country.to_csv("valid_country_output.csv")
    valid_country = valid_country.values
    print("Valid country: %s"% (len(valid_country)))

    # assumo che la riga tutte le righe siano valide per diminuire il numero di iterazioni
    df_valid['is_valid'] = True
    
    for index, row in df_valid.iterrows():
        if (row.home_team not in valid_country and row.away_team not in valid_country):
            df_valid.at[index, 'is_valid'] = False

    return df_valid

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

def convert_onehot(home_team, away_team, tournament='Friendly', city='Rome', country='Italy', continent='Europe', neutral=0): # = 1 True = 0 False
    # load dataframes ...

    
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