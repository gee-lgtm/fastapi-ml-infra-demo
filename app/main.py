from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import time

from prometheus_client import Counter, Histogram, generate_latest, REGISTRY

app = FastAPI(title="ML Infra Demo API")

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
