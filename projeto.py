import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

dados = pd.read_csv("wholesale.csv")
modelo = DecisionTreeClassifier(random_state=42)

print(dados.head())

dados["Channel"] = dados["Channel"].replace({"HoReCa": 0, "Retail": 1}).astype("int64")

dados["Region"] = (
    dados["Region"].replace({"Lisbon": 0, "Oporto": 1, "Other": 2}).astype("int64")
)

print(
    "----------------------------------------------------------------------------------------------"
)
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
    "Channel",
]

dados = dados.reindex(columns=nova_ordem)

print(
    "----------------------------------------------------------------------------------------------"
)
print(dados.head())

X = dados.drop(columns=["Channel"])

y = dados["Channel"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Amostras para treinamento:", len(X_treino))
print("Amostras para teste:", len(X_teste))


modelo.fit(X_treino, y_treino)
y_previsto = modelo.predict(X_teste)


print(
    "----------------------------------------------------------------------------------------------"
)
print(classification_report(y_teste, y_previsto, target_names=["HoReCa", "Retail"]))


regioes = {"Lisbon": 0, "Oporto": 1, "Other": 2}

while True:
    print(
        "----------------------------------------------------------------------------------------------"
    )
    print("Classificar um novo cliente")

    regiao_texto = input("Região (Lisbon, Oporto ou Other): ").strip().capitalize()
    if regiao_texto not in regioes:
        print("Região inválida. Use Lisbon, Oporto ou Other.")
        continue

    try:
        fresh = float(input("Gasto anual com Fresh: "))
        milk = float(input("Gasto anual com Milk: "))
        grocery = float(input("Gasto anual com Grocery: "))
        frozen = float(input("Gasto anual com Frozen: "))
        detergents = float(input("Gasto anual com Detergents_Paper: "))
        delicatessen = float(input("Gasto anual com Delicatessen: "))
    except ValueError:
        print("Valor inválido. Digite apenas números.")
        continue

    regiao = regioes[regiao_texto]

    nova_amostra = pd.DataFrame(
        [[regiao, fresh, milk, grocery, frozen, detergents, delicatessen]],
        columns=X.columns,
    )

    resultado = modelo.predict(nova_amostra)[0]

    if resultado == 0:
        print("Canal previsto: HoReCa")
    else:
        print("Canal previsto: Retail")

    continuar = input("Classificar outro cliente? (s/n): ")
    if continuar.strip().lower() != "s":
        break
