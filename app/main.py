from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging
import traceback
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.core.settings import settings
from app.routers.pages.home import router as home_router
from app.routers.api.health import router as health_router
from app.routers.api.admin import router as admin_router
from app.routers.api.reco import router as reco_router


def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME)

    # Dev-only: return full traceback as JSON to speed up debugging
    if str(getattr(settings, "ENV", "")).lower() in {"local", "dev", "development"}:
        logger = logging.getLogger("uvicorn.error")

        @app.exception_handler(Exception)
        async def _unhandled_exception_handler(request: Request, exc: Exception):
            logger.exception("Unhandled exception: %s %s", request.method, request.url)
            return JSONResponse(
                status_code=500,
                content={
                    "ok": False,
                    "error": str(exc),
                    "type": exc.__class__.__name__,
                    "path": str(request.url),
                    "trace": traceback.format_exc(),
                },
            )

    # Templates
    templates = Jinja2Templates(directory=str(settings.TEMPLATE_DIR))
    app.state.templates = templates  # pages 라우터에서 사용

    # Static
    app.mount(
        "/static",
        StaticFiles(directory=str(settings.STATIC_DIR)),
        name="static",
    )

    # Routers
    app.include_router(home_router)
    app.include_router(health_router)
    app.include_router(admin_router)
    app.include_router(reco_router)
    return app


app = create_app()