# %%

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection, tree
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
# SAMPLE

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

X_train.isna().sum().sort_values(ascending=False)
# %%

df_analise = X_train.copy()
df_analise[target] = y_train
sumario = df_analise.groupby(by=target).agg(["mean", "median"]).T
pd.set_option('display.max_rows', 100)
sumario
# %%

sumario['diferenca_abs'] = sumario[0] - sumario[1]
sumario['diferenca_rel'] = sumario[0] / sumario[1]
sumario.sort_values(by=['diferenca_rel'], ascending=False)
# %%

X_train.head()
# %%

arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X_train, y_train)
# %%

feature_importances = (pd.Series(arvore.feature_importances_, index=X_train.columns)
                        .sort_values(ascending=False)
                        .reset_index())


feature_importances['acum']=feature_importances[0].cumsum()
feature_importances[feature_importances[0] > 0.01] # Selecionar pelo atributo com importancia maior do que 1%
#feature_importances[feature_importances['acum'] < 0.96] # Selecionar pelo atributo com importancia acumulada até 95%
# %%
