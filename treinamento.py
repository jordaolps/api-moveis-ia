import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

print("1. Carregando os dados...")
# Carrega o arquivo baixado do Kaggle
df = pd.read_csv('data/train.csv')

# Vamos escolher apenas 6 características principais para não complicar o formulário do front-end
# OverallQual: Qualidade geral do material e acabamento (1 a 10)
# GrLivArea: Área construída acima do solo (em pés quadrados)
# GarageCars: Tamanho da garagem (capacidade de carros)
# TotalBsmtSF: Área total do porão
# FullBath: Banheiros completos
# YearBuilt: Ano de construção original

features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
target = 'SalePrice'

# Separando os dados (X = características, y = preço)
X = df[features]
y = df[target]

print("2. Tratando dados nulos...")
# Preenchendo valores vazios com zero (caso existam)
X = X.fillna(0)

print("3. Separando dados de treino e teste...")
# Separamos 20% dos dados para testar se o modelo aprendeu mesmo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("4. Treinando o modelo de Inteligência Artificial...")
# Usaremos o Random Forest (Floresta Aleatória), que é excelente para esse tipo de dado
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

print("5. Avaliando o modelo...")
# Fazendo previsões com os 20% que separamos
previsoes = modelo.predict(X_test)

# Calculando a margem de erro
erro_medio = mean_absolute_error(y_test, previsoes)
precisao = r2_score(y_test, previsoes) * 100

print(f"-> Erro Médio Absoluto: ${erro_medio:,.2f}")
print(f"-> Precisão do Modelo (R²): {precisao:.2f}%")

print("6. Salvando o modelo para usar no Back-end...")
# Salva o modelo treinado em um arquivo .pkl
joblib.dump(modelo, 'modelo_preco_imoveis.pkl')
print("✅ Arquivo 'modelo_preco_imoveis.pkl' criado com sucesso!")