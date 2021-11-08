lt.rcParams.update({'figure.figsize':(13,5), 'figure.dpi':100})
# prima mi devo trovare la distribuzione che più somiglia alla mia

df = pd.read_csv(path.join(PATH_DST, 'dataset_v3_zscore.csv')).drop(columns=['Unnamed: 0'])
neutral_games = df.query("neutral == True")
not_neutral_games = df.query("neutral == False")

plt.hist(neutral_games[['home_score', 'away_score']].values, range(6), alpha=0.8, label=['Home_neutral', 'Away_neutral'],density=True, color=["#6fa8dc", "#f6b26b"])
plt.hist(not_neutral_games[['home_score', 'away_score']].values, range(6), alpha=0.8, label=['Home_not_neutral', 'Away_not_neutral'],density=True, color=["#0b5394", "#b45f06"])

means_neutral = neutral_games[['home_score','away_score']].mean()
means_not_neutral = not_neutral_games[['home_score','away_score']].mean()

# construct Poisson  for each mean goals value
poisson_pred = np.column_stack([[poisson.pmf(k, means_neutral[j]) for k in range(10)] for j in range(2)])
poisson_pred_not_neutral = np.column_stack([[poisson.pmf(k, means_not_neutral[j]) for k in range(10)] for j in range(2)])

# add lines for the Poisson distributions
pois1, = plt.plot([i-0.5 for i in range(1,11)], poisson_pred[:,0],linestyle='-', marker='o',label="Home_neutral", color = '#6fa8dc')
pois2, = plt.plot([i-0.5 for i in range(1,11)], poisson_pred[:,1],linestyle='-', marker='o',label="Away_neutral", color = '#f6b26b')

pois1, = plt.plot([i-0.5 for i in range(1,11)], poisson_pred_not_neutral[:,0],linestyle='-', marker='o',label="Home_not_neutral", color = '#0b5394')
pois2, = plt.plot([i-0.5 for i in range(1,11)], poisson_pred_not_neutral[:,1],linestyle='-', marker='o',label="Away_not_neutral", color = '#b45f06')

leg=plt.legend(loc='upper right', fontsize=16, ncol=2)
plt.xlabel("Goals per Match",size=18)
plt.ylabel("Matches (density=True)",size=18)
plt.title("Result per match",size=20,fontweight='bold')
plt.show()