import os
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

    embeddings = [np.array(x.embedding, dtype=np.float32)
                  for x in response.data]

    query_embedding = embeddings[0]

    candidate_embeddings = embeddings[1:]

    query_embedding /= np.linalg.norm(query_embedding)

    scores = []

    for i, emb in enumerate(candidate_embeddings):

        emb /= np.linalg.norm(emb)

        score = float(np.dot(query_embedding, emb))

        scores.append((score, i))

    scores.sort(reverse=True)

    ranking = [i for _, i in scores[:3]]

    return {"ranking": ranking}
