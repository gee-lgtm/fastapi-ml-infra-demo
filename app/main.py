import asyncio
import logging
import time
from dataclasses import dataclass
from functools import partial
from typing import List

import anyio
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------
logger = logging.getLogger("uvicorn.error")

# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------
app = FastAPI(title="ML Infra Demo - Batched Embeddings")

# -------------------------------------------------------------------
# Metrics (Prometheus)
# -------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "Latency of HTTP requests in seconds",
    ["method", "path"],
)

EMBED_BATCH_SIZE = Histogram(
    "embed_batch_size",
    "Number of texts processed per embed batch",
)

# -------------------------------------------------------------------
# Models (Pydantic)
# -------------------------------------------------------------------
class EmbedRequest(BaseModel):
    texts: List[str]


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]


# -------------------------------------------------------------------
# SentenceTransformers model (global)
# -------------------------------------------------------------------
logger.info("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
logger.info("Model loaded.")


# -------------------------------------------------------------------
# Batching infrastructure
# -------------------------------------------------------------------
MAX_BATCH_SIZE = 16          # max requests per batch (tweak as you like)
MAX_BATCH_DELAY = 0.01       # seconds to wait to accumulate a batch

@dataclass
class EmbedJob:
    texts: List[str]
    future: asyncio.Future  # will hold List[List[float]]


request_queue: "asyncio.Queue[EmbedJob]" = asyncio.Queue()


async def batch_worker():
    """
    Background task that:
    - pulls EmbedJobs from the queue
    - groups them into batches
    - runs a single model.encode on the combined texts
    - splits the result and fulfills each job.future
    """
    logger.info("Batch worker started")
    while True:
        try:
            # Wait for at least one job
            first_job = await request_queue.get()
            batch = [first_job]
            start = time.monotonic()

            # Try to accumulate more jobs up to MAX_BATCH_SIZE or MAX_BATCH_DELAY
            while len(batch) < MAX_BATCH_SIZE:
                remaining = MAX_BATCH_DELAY - (time.monotonic() - start)
                if remaining <= 0:
                    break

                try:
                    job = await asyncio.wait_for(request_queue.get(), timeout=remaining)
                    batch.append(job)
                except asyncio.TimeoutError:
                    break

            # Flatten all texts into one list
            all_texts: List[str] = []
            for job in batch:
                all_texts.extend(job.texts)

            logger.info(f"Processing batch of {len(batch)} jobs, {len(all_texts)} texts total")
            EMBED_BATCH_SIZE.observe(len(all_texts))

            # Run model.encode in a worker thread to avoid blocking the event loop
            encode_fn = partial(model.encode, convert_to_numpy=True)
            embeddings = await anyio.to_thread.run_sync(
                encode_fn,
                all_texts,
            )
            embeddings = embeddings.tolist()

            # Split embeddings back per job
            idx = 0
            for job in batch:
                n = len(job.texts)
                job_embeds = embeddings[idx : idx + n]
                idx += n
                if not job.future.done():
                    job.future.set_result(job_embeds)
                request_queue.task_done()

        except Exception as e:
            logger.exception(f"Error in batch_worker: {e}")
            # If we hit an exception, make sure we don't leave futures hanging
            # (In real prod code, we might retry or mark jobs failed)
            continue


@app.on_event("startup")
async def startup_event():
    # Kick off the batching worker in the background
    asyncio.create_task(batch_worker())
    logger.info("Startup complete, background batch worker running")


# -------------------------------------------------------------------
# Middleware for request metrics
# -------------------------------------------------------------------
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    path = request.url.path
    method = request.method

    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        status = 500
        raise
    finally:
        elapsed = time.perf_counter() - start_time
        REQUEST_COUNT.labels(method=method, path=path, status=status).inc()
        REQUEST_LATENCY.labels(method=method, path=path).observe(elapsed)

    return response


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------
@app.get("/health")
def health():
    """
    Simple health check used by Kubernetes liveness/readiness probes.
    """
    logger.info("Health check OK")
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    """
    Prometheus scrape endpoint.
    """
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """
    Takes a list of texts and returns their embeddings.
    Requests are batched internally by the background worker.
    """
    if not request.texts:
        return EmbedResponse(embeddings=[])

    loop = asyncio.get_event_loop()
    fut: asyncio.Future = loop.create_future()
    job = EmbedJob(texts=request.texts, future=fut)

    await request_queue.put(job)
    embeddings: List[List[float]] = await fut

    return EmbedResponse(embeddings=embeddings)
