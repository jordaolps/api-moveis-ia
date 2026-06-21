from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware # Adicionado para permitir o Front-end depois!

# Carrega as variáveis do arquivo .env
load_dotenv()

# Inicializando a API
app = FastAPI(title="API - Previsão de Imóveis e Consultoria IA")

# --- NOVIDADE: Configuração de CORS (Permite que o Front-end acesse essa API) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Depois podemos restringir só para o link do front-end
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carregando o modelo treinado
modelo = joblib.load('modelo_preco_imoveis.pkl')

# Configurando o Google Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
modelo_ia = genai.GenerativeModel('gemini-3.1-flash-lite')

# --- MOLDES E ROTA 1 (/prever) CONTINUAM IGUAIS AQUI ---
class Imovel(BaseModel):
    OverallQual: int
    GrLivArea: int
    GarageCars: int
    TotalBsmtSF: int
    FullBath: int
    YearBuilt: int

class MensagemChat(BaseModel):
    preco_estimado: float
    pergunta: str
    detalhes_imovel: str

@app.post("/prever")
def prever_preco(imovel: Imovel):
    dados_entrada = np.array([[imovel.OverallQual, imovel.GrLivArea, imovel.GarageCars, imovel.TotalBsmtSF, imovel.FullBath, imovel.YearBuilt]])
    preco_previsto = modelo.predict(dados_entrada)[0]
    return {"preco_estimado": round(preco_previsto, 2)}

# --- ROTA 2 ATUALIZADA (LIGADA AO GEMINI REAL) ---
@app.post("/consultar-ia")
def consultar_ia(mensagem: MensagemChat):
    # O Prompt Mágico
    prompt = f"""
    Você é um corretor de imóveis experiente, simpático e realista. 
    O usuário está avaliando o seguinte imóvel: {mensagem.detalhes_imovel}.
    O nosso modelo de Machine Learning estimou o valor dessa casa em ${mensagem.preco_estimado:,.2f}.
    
    O usuário te perguntou: "{mensagem.pergunta}"
    
    Responda de forma direta, profissional, em português do Brasil, e ajude o usuário com a dúvida dele usando o preço estimado como base. Seja breve (no máximo 2 parágrafos).
    """
    
    # Chamando a IA do Google
    resposta = modelo_ia.generate_content(prompt)
    
    return {"resposta": resposta.text}