# %%

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection, tree, linear_model, metrics, pipeline
from feature_engine import discretisation, encoding
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
# EXPLORE

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

best_features = (feature_importances[feature_importances['acum'] < 0.96]['index']
                 .tolist())

best_features
# %%
# MODIFY

# Discretizar
tree_discretization = discretisation.DecisionTreeDiscretiser(variables=best_features,
                                                             regression=False,
                                                             bin_output='bin_number',
                                                             cv=3)

# OneHotEncoding
onehot = encoding.OneHotEncoder(variables=best_features, ignore_format=True)

# %%
# MODEL

reg = linear_model.LogisticRegression(penalty=None, random_state=42, max_iter=100000)

model_pipeline = pipeline.Pipeline(
    steps=[
        ('Discretizar', tree_discretization),
        ('Onehot', onehot),
        ('Model', reg),
    ]
)

model_pipeline.fit(X_train, y_train)
# %%

y_train_predict = model_pipeline.predict(X_train)
y_train_prob = model_pipeline.predict_proba(X_train)[:,1]

acc_train = metrics.accuracy_score(y_train, y_train_predict)
auc_train = metrics.roc_auc_score(y_train, y_train_prob)

print("Acurárcia Treino:", acc_train)
print("AUC Treino:", auc_train)
# %%

y_test_predict = model_pipeline.predict(X_test)
y_test_prob = model_pipeline.predict_proba(X_test)[:,1]

acc_test = metrics.accuracy_score(y_test, y_test_predict)
auc_test = metrics.roc_auc_score(y_test, y_test_prob)

print("Acurárcia Teste:", acc_test)
print("AUC Teste:", auc_test)
# %%
