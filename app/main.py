from fastapi import FastAPI

from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Support Automation System")


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("Application startup", extra={"event": "startup", "app_env": settings.app_env})


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
