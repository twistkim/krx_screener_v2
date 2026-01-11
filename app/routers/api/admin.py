from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession


from app.core.db import Base, engine, get_db
from app.services.ingest_service import ingest_missing
from app.services.screener_service import run_screen

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/init-db")
async def init_db():
    # IMPORTANT: make sure all ORM models are imported so Base.metadata knows every table.
    # (Models are split across multiple files; importing them here registers tables once.)
    import app.models.symbol  # noqa: F401
    import app.models.daily_bar  # noqa: F401
    import app.models.screen_run  # noqa: F401
    import app.models.tables  # noqa: F401  # (e.g., Recommendation)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return {"ok": True}


@router.post("/ingest")
async def admin_ingest(
    market: str = Query(default="ALL"),
    days: int = Query(default=60, ge=1, le=5000),
    asof: str | None = Query(default=None),  # yyyymmdd
    start: str | None = Query(default=None),  # yyyy-mm-dd or yyyymmdd
    end: str | None = Query(default=None),  # yyyy-mm-dd or yyyymmdd
    db: AsyncSession = Depends(get_db),
):
    return await ingest_missing(db, market=market, days=days, asof=asof, start=start, end=end)


@router.post("/screen")
async def admin_screen(
    market: str = Query(default="ALL"),
    top_n: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    return await run_screen(db, market=market, top_n=top_n)