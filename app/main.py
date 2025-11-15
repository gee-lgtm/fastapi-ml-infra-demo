from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import time
import logging

from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
from sentence_transformers import SentenceTransformer
from typing import List

app = FastAPI(title="ML Infra Demo API")
logger = logging.getLogger("uvicorn.error")
logger.info("Loading sentence-transformers model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
logger.info("Model loaded.")

# ---- Prometheus metrics ----
REQUEST_COUNT = Counter(
    "app_requests_total",
    "Total number of requests",
    ["method", "endpoint", "http_status"],
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
)

class HealthResponse(BaseModel):
    status: str
    timestamp: float

class EchoRequest(BaseModel):
    message: str

class EchoResponse(BaseModel):
    message: str
    length: int
    processed_at: float

class EmbedRequest(BaseModel):
    texts: List[str]

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    endpoint = request.url.path
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(process_time)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        http_status=str(response.status_code),
    ).inc()

    return response

@app.get("/health", response_model=HealthResponse)
def health():
    logger.info("Health check OK")
    return HealthResponse(status="ok", timestamp=time.time())

@app.post("/echo", response_model=EchoResponse)
def echo(request: EchoRequest):
    return EchoResponse(
        message=request.message.upper(),
        length=len(request.message),
        processed_at=time.time(),
    )

@app.get("/metrics")
def metrics():
    # Expose Prometheus metrics in plaintext format
    data = generate_latest(REGISTRY)
    return PlainTextResponse(content=data.decode("utf-8"), media_type="text/plain")

@app.post("/embed")
def embed(request: EmbedRequest):
    """
    Takes a list of texts and returns their embeddings.
    """
    logger.info(f"Embedding {len(request.texts)} texts")
    emb = model.encode(request.texts, convert_to_numpy=True).tolist()
    # returns a vector like [0.01, -0.12, ...]
    return EmbedResponse(embeddings=emb)

