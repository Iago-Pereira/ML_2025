# %%

import pandas as pd
from sklearn import model_selection
# %%

df = pd.read_csv("../data/abt_churn.csv")
df.head()
# %%

out = df[df["dtRef"]==df['dtRef'].max()].copy()
out
# %%

df_train = df[df["dtRef"]<df['dtRef'].max()].copy()
# %%

features = df_train.columns[2:-1]

target = 'flagChurn'

X, y = df_train[features], df_train[target]
# %%

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                    random_state=42,
                                                                    test_size=0.2,
                                                                    stratify=y #Para melhorar taxa variável resposta
                                                                    )
# %%

# Verificando taxa da variável resposta do target
print(f"Taxa variável resposta Treino: {y_train.mean()}")
print(f"Taxa variável resposta Teste: {y_test.mean()}")
# %%
