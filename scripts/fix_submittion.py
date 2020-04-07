import pandas as pd

df = pd.read_csv('../models/submission_cat_ii.csv')
df.id = df.id.astype(int)
df.to_csv('../models/submission_cat_ii.csv', index=False)