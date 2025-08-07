# %%
import pandas as pd

model_df = pd.read_pickle("model.pkl")
model = model_df['model']
features = model_df['features']

# %%
model_df
# %%

df = pd.read_csv("../data/abt_churn.csv")
amostra = df[df['dtRef'] == df['dtRef'].max()].sample(3)
amostra = amostra.drop('flagChurn', axis=1)
# %%

predicao = model.predict_proba(amostra[features])[:,1]
amostra['proba'] = predicao
amostra
# %%
