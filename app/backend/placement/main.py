from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import api, config, db


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_content()
    yield


app = FastAPI(title="ML coding learning site", lifespan=lifespan)
config.MEDIA_DIR.mkdir(exist_ok=True)
app.include_router(api.router)
app.mount("/media", StaticFiles(directory=config.MEDIA_DIR), name="media")


@app.get("/api/health")
def health():
    return {"status": "ok", "exercises_db": config.EXERCISES_DB.exists(), "content_db": config.CONTENT_DB.exists()}
