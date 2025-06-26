# %%

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
# %%

df_clones = pd.read_parquet("../data/dados_clones.parquet")
df_clones.head(10)
# %%

df_clones.columns = df_clones.columns.str.replace(' ','_', regex=False)
df_clones = df_clones.rename(columns={'Status_': 'Status'})
df_clones.head(10)
# %%

features = ['Massa(em_kilos)', 'Estatura(cm)']
target = 'Status'

# %%

print(df_clones['General_Jedi_encarregado'].unique())
print(df_clones['Distância_Ombro_a_ombro'].unique())
print(df_clones['Tamanho_do_crânio'].unique())
print(df_clones['Tamanho_dos_pés'].unique())
# %%

X = df_clones[features]
y = df_clones[target]

# Generais não foram considerados para o treinamento
X = X.replace({
    'Yoda': 1, 'Shaak Ti': 2, 'Obi-Wan Kenobi': 3, 'Aayla Secura': 4, 'Mace Windu': 5,
    'Tipo 1': 1, 'Tipo 2': 2, 'Tipo 3': 3, 'Tipo 4': 4, 'Tipo 5': 5,
})

X
# %%

model = tree.DecisionTreeClassifier()
model.fit(X=X, y=y)
# %%

plt.figure(dpi=400)

tree.plot_tree(model, feature_names=features,
               class_names=model.classes_,
               filled=True, max_depth=3
               )
# %%
