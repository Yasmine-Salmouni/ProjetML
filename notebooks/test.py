# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Test de conversion avec Jupytext
# Ce notebook est utilisé pour valider le processus de synchronisation `.ipynb` → `.py`.

# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Création d'un DataFrame fictif
df = pd.DataFrame({
    'x': range(10),
    'y': [i**2 for i in range(10)]
})
df.head()

# %%
# Visualisation simple
plt.plot(df['x'], df['y'])
plt.title('Courbe quadratique')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
