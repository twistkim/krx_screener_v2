# app/routers/api/reco.py
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.core.db import get_db

router = APIRouter(prefix="/api/reco", tags=["reco"])


@router.get("")
async def get_reco(
    run_id: int = Query(..., ge=1),
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    q = text(
        """
        SELECT r.run_id, r.symbol, r.score, r.meta_json, s.name, s.market
        FROM recommendations r
        LEFT JOIN symbols s ON BINARY s.symbol = BINARY r.symbol
        WHERE r.run_id=:rid
        ORDER BY r.score DESC
        LIMIT :lim
        """
    )
    rows = (await db.execute(q, {"rid": run_id, "lim": limit})).mappings().all()
    return {"ok": True, "run_id": run_id, "items": [dict(r) for r in rows]}


@router.get("/latest")
async def get_reco_latest(
    market: str = Query(default="ALL"),
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    m = (market or "ALL").upper().strip()

    if m in ("ALL", "*"):
        q_run = text(
            """
            SELECT id, asof_date, market
            FROM screen_runs
            ORDER BY id DESC
            LIMIT 1
            """
        )
        run = (await db.execute(q_run)).mappings().first()
    else:
        q_run = text(
            """
            SELECT id, asof_date, market
            FROM screen_runs
            WHERE market=:m
            ORDER BY id DESC
            LIMIT 1
            """
        )
        run = (await db.execute(q_run, {"m": m})).mappings().first()

    if not run:
        return {"ok": True, "run_id": None, "market": m, "items": []}

    q = text(
        """
        SELECT r.run_id, r.symbol, r.score, r.meta_json, s.name, s.market
        FROM recommendations r
        LEFT JOIN symbols s ON BINARY s.symbol = BINARY r.symbol
        WHERE r.run_id=:rid
        ORDER BY r.score DESC
        LIMIT :lim
        """
    )
    rows = (await db.execute(q, {"rid": run["id"], "lim": limit})).mappings().all()
    return {
        "ok": True,
        "run_id": run["id"],
        "asof_date": run["asof_date"],
        "market": run["market"],
        "items": [dict(r) for r in rows],
    }