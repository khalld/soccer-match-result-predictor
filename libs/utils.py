from numba import jit # import decorator that allow to use gpu
from timeit import default_timer as timer   

@jit
def find_penalty(row, sht_csv, sht_csv_len):
    for i in range (0, sht_csv_len):

        if( row['date'] == sht_csv.iloc[i]['date'] and row['home_team'] == sht_csv.iloc[i]['home_team'] and row['away_team'] == sht_csv.iloc[i]['away_team'] ) is True:

            if(row['home_team'] == sht_csv.iloc[i]['winner']):
                return 'D-HP'

            if(row['away_team'] == sht_csv.iloc[i]['winner']):
                return 'D-AP'

    return 'D'

@jit(nopython=True)
def return_outcome(home_score,away_score):
    if (home_score > away_score):
        return 'Home'
    if (away_score > home_score):
        return 'Away'
    if (home_score == away_score):
        return 'Draw'


@jit
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

# @jit(nopython=True)
def find_range(year):
    arr = ['1870-1879', '1880-1889', '1890-1899', '1900-1909', '1910-1919', '1920-1929', '1930-1939', '1940-1949', '1950-1959', '1960-1969', '1970-1979', '1980-1989', '1990-1999', '2000-2009', '2010-2020','2021-2029']

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