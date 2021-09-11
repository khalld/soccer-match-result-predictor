from numba import jit # import decorator that allow to use gpu

@jit
def find_penalty(row, sht_csv, sht_csv_len):
    for i in range (0, sht_csv_len):

        if( row['date'] == sht_csv.iloc[i]['date'] and row['home_team'] == sht_csv.iloc[i]['home_team'] and row['away_team'] == sht_csv.iloc[i]['away_team'] ) is True:

            if(row['home_team'] == sht_csv.iloc[i]['winner']):
                return 'HP'

            if(row['away_team'] == sht_csv.iloc[i]['winner']):
                return 'AP'

    return 'D'

@jit
def return_outcome(home_score,away_score):
    if (home_score > away_score):
        return 'H'
    if (away_score > home_score):
        return 'A'
    if (home_score == away_score):
        return 'D'


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