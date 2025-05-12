# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# CORS ayarı (WordPress frontend için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # yayına alırken domain ile sınırla!
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    keyword: str
    text: str

@app.post("/analyze")
async def analyze_similarity(data: InputData):
    def get_embedding(text):
        res = openai.Embedding.create(input=text, model="text-embedding-3-small")
        return res["data"][0]["embedding"]

    try:
        keyword_vec = get_embedding(data.keyword)
        text_vec = get_embedding(data.text)
        similarity = cosine_similarity([keyword_vec], [text_vec])[0][0]
        return { "similarity": round(similarity, 4) }
    except Exception as e:
        return { "error": str(e) }
