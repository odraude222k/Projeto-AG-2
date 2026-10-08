import pandas as pd

dados = pd.read_csv("wholesale.csv")

dados["Channel"] = dados["Channel"].replace({
    "HoReCa": 0,
    "Retail": 1
})

dados["Region"] = dados["Region"].replace({
    "Lisbon": 0,
    "Oporto": 1,
    "Other": 2
})

print(dados.head())