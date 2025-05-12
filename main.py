from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# OpenAI istemcisi (yeni v1 yapısı)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FastAPI uygulaması
app = FastAPI()

# CORS ayarı – slash olmadan domain tanımla
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.batuhandurmaz.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Girdi modeli
class InputData(BaseModel):
    keyword: str
    text: str

# Vektör (embedding) alma fonksiyonu
def get_embedding(text: str):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Benzerlik hesaplama endpoint'i
@app.post("/analyze")
async def analyze_similarity(data: InputData):
    try:
        keyword_vec = get_embedding(data.keyword)
        text_vec = get_embedding(data.text)
        similarity = cosine_similarity([keyword_vec], [text_vec])[0][0]
        return { "similarity": round(similarity, 4) }
    except Exception as e:
        return { "error": str(e) }
