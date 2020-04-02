import pandas as pd

df = pd.read_csv('models/randomforest.csv')
df.id = df.id.astype(int)
df.to_csv('models/rf_fixed.csv', index=False)