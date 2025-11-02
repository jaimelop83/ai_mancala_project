# Exploration of fitness log and training statistics CSV files

import pandas as pd

fitness_log = pd.read_csv('recent_models/fitness_log.csv')
training_stats = pd.read_csv('recent_models/trainingStats.csv')

print(fitness_log.head())
print(training_stats.head())