import pandas as pd
from sklearn.model_selection import train_test_split


dados = pd.read_csv("wholesale.csv")

print(dados.head())

dados["Channel"] = dados["Channel"].replace({
    "HoReCa": 0,
    "Retail": 1
})

dados["Region"] = dados["Region"].replace({
    "Lisbon": 0,
    "Oporto": 1,
    "Other": 2
})

print("----------------------------------------------------------------------------------------------")
print(dados.head())
print(dados.dtypes)

nova_ordem = [
    "Region",
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicatessen",
    "Channel"
]

dados = dados.reindex(columns=nova_ordem)

print("----------------------------------------------------------------------------------------------")
print(dados.head())

X = dados.drop(columns=["Channel"])

y = dados["Channel"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Amostras para treinamento:", len(X_treino))
print("Amostras para teste:", len(X_teste))

