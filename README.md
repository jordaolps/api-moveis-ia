# API de Previsão Imobiliária + Consultoria com IA (Google Gemini)

Este repositório contém o Back-end do projeto de previsão de preços de imóveis, desenvolvido como parte do meu portfólio Full-Stack e Data Science. 

A aplicação une **Machine Learning** (para prever o valor de uma casa) e **Inteligência Artificial Generativa** (para atuar como um corretor virtual), fornecendo uma API rápida e robusta construída em **FastAPI**.

## Tecnologias Utilizadas
* **Python 3**
* **FastAPI** (Construção da API e documentação automática)
* **Scikit-Learn** (Treinamento do modelo de Machine Learning)
* **Google Generative AI (Gemini 3.1 Flash Lite)** (LLM para o consultor virtual)
* **Uvicorn** (Servidor ASGI)
* **Pandas & NumPy** (Tratamento e manipulação de dados)

## Arquitetura e Modelagem de Dados
O modelo preditivo foi treinado utilizando o famoso dataset *House Prices - Advanced Regression Techniques* (Kaggle).
1. **Seleção de Features:** Foram selecionadas as 6 variáveis de maior impacto: `OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`, `FullBath`, `YearBuilt`.
2. **Algoritmo:** Utilizamos o `RandomForestRegressor`.
3. **Performance:** O modelo alcançou uma **Precisão (R²) de 89.08%** nos dados de teste, sendo exportado em formato `.pkl` para consumo pela API.
4. **Engenharia de Prompt:** O preço calculado matematicamente é injetado como contexto no prompt do Google Gemini, garantindo que o LLM não gere "alucinações" sobre os valores, fornecendo uma consultoria realista.

## Como rodar o projeto localmente

1. Clone este repositório: ```git clone https://github.com/jordaolps/api-imoveis-ia.git```
   
    ```cd api-imoveis-ia```

2. Instale as dependências: ```pip install -r requirements.txt```

3. Configure a chave da API:
   
Crie um arquivo chamado .env na raiz do projeto e adicione sua chave do Google Gemini:

```GEMINI_API_KEY=sua_chave_aqui```

4. Inicie o servidor:

    ```uvicorn main:app --reload```

No caso desse projeto a API está rodando no Render, mas para rodar localmente utilize: http://127.0.0.1:8000. 
Você pode acessar a documentação interativa (Swagger UI) em http://127.0.0.1:8000/docs.

## Rotas da API
- `POST /prever:` Recebe as características do imóvel (JSON) e retorna o preco_estimado calculado pelo modelo .pkl.
- `POST /consultar-ia:` Recebe o preço calculado e a dúvida do usuário, retornando uma resposta contextualizada do Gemini.