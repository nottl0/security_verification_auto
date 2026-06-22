from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.analyzer import analyze_offense
from app.models import AISummary, Offense
from app.vector_store import build_vector_store

@asynccontextmanager
async def lifespan(app: FastAPI):
    build_vector_store()
    yield 

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze", response_model=AISummary)
def analyze(offense: Offense):
    return analyze_offense(offense)