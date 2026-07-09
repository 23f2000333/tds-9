import os
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# AI Pipe OpenAI-compatible endpoint
client = OpenAI(
    api_key=os.getenv("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1"
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RankRequest(BaseModel):
    query_id: str
    query: str
    candidates: list[str]


@app.post("/rank")
def rank(req: RankRequest):

    texts = [req.query] + req.candidates

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    embeddings = [
        np.asarray(x.embedding, dtype=np.float32)
        for x in response.data
    ]

    query_embedding = embeddings[0]
    candidate_embeddings = embeddings[1:]

    # Normalize query once
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    scores = []

    for idx, emb in enumerate(candidate_embeddings):
        emb = emb / np.linalg.norm(emb)
        score = float(np.dot(query_embedding, emb))
        scores.append((score, idx))

    # Highest cosine similarity first
    scores.sort(key=lambda x: x[0], reverse=True)

    top3 = [idx for _, idx in scores[:3]]

    return {"ranking": top3}
