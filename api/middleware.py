import time
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("sentenialx.api")
logging.basicConfig(level=logging.INFO)

def setup_middleware(app: FastAPI) -> None:
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten in prod
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def timing_middleware(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        dur_ms = (time.time() - start) * 1000
        logger.info("%s %s -> %s in %.2fms",
                    request.method, request.url.path, response.status_code, dur_ms)
        response.headers["X-Response-Time-ms"] = f"{dur_ms:.2f}"
        return response
