# %%

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, tree
# %%

df = pd.read_excel("../data/dados_cerveja_nota.xlsx")
df.head()
# %%

X = df[['cerveja']] # Isso é uma matriz (dataframe)
y = df['nota']      # Isso é um vetor (series)

# Aprendizado de máquina
reg = linear_model.LinearRegression(fit_intercept=True)
reg.fit(X, y)
# %%

# O melhor ajuste da regressão linear
a, b = reg.intercept_, reg.coef_[0]
print(a, b)
# %%

predict_reg = reg.predict(X.drop_duplicates())
# %%

arvore_full = tree.DecisionTreeRegressor(random_state=42)
arvore_full.fit(X, y)
predict_arvore_full = arvore_full.predict(X.drop_duplicates())

# %%

arvore_d2 = tree.DecisionTreeRegressor(random_state=42, max_depth= 2)
arvore_d2.fit(X, y)
predict_arvore_d2 = arvore_d2.predict(X.drop_duplicates())
# %%

plt.plot(X['cerveja'], y, 'o')
plt.grid(True)
plt.title("Relação Cerveja vs Nota")
plt.xlabel("Cerveja")
plt.ylabel("Nota")
plt.plot(X.drop_duplicates()['cerveja'], predict_reg) #reta regressao linear
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_full) # linhas árvore de decisão
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_d2) # linhas árvore de decisão
plt.legend(['Observado', f'y = {a:.3f} + {b:.3f} x',
            'Árvore Full',
            'Árvore Depth = 2',
            ])
# %%

plt.figure(dpi=400)

tree.plot_tree(arvore_d2,
               feature_names=['cerveja'],
               filled=True)
# %%
