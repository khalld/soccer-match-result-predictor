def find_penalty(row, sht_csv, sht_csv_len):
    for i in range (0, sht_csv_len):

        if( row['date'] == sht_csv.iloc[i]['date'] and row['home_team'] == sht_csv.iloc[i]['home_team'] and row['away_team'] == sht_csv.iloc[i]['away_team'] ) is True:

            if(row['home_team'] == sht_csv.iloc[i]['winner']):
                return 'HP'

            if(row['away_team'] == sht_csv.iloc[i]['winner']):
                return 'AP'

    return 'D'

def return_outcome(home_score,away_score):
    if (home_score > away_score):
        return 'H'
    if (away_score > home_score):
        return 'A'
    if (home_score == away_score):
        return 'D'