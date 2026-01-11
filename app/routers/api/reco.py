# app/routers/api/reco.py
from __future__ import annotations

import json
from decimal import Decimal
from typing import Any, Dict

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.core.db import get_db

router = APIRouter(prefix="/api/reco", tags=["reco"])


def _parse_meta(meta_json: Any) -> Dict[str, Any] | None:
    """DB의 meta_json을 프론트에서 쓰기 쉬운 dict로 변환."""
    if meta_json is None:
        return None

    # MySQL JSON 컬럼은 드라이버/설정에 따라 dict/str/bytes로 올 수 있음
    if isinstance(meta_json, dict):
        return meta_json

    if isinstance(meta_json, (bytes, bytearray)):
        try:
            meta_json = meta_json.decode("utf-8")
        except Exception:
            meta_json = str(meta_json)

    if isinstance(meta_json, str):
        s = meta_json.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            # 파싱 실패 시 raw를 그대로 담아 반환
            return {"_raw": meta_json}

    # 그 외 타입
    return {"_raw": str(meta_json)}


def _norm_row(r: Dict[str, Any]) -> Dict[str, Any]:
    """API 응답을 안정적으로 직렬화 가능한 형태로 정규화."""
    score = r.get("score")
    if isinstance(score, Decimal):
        score = float(score)

    meta_json = r.get("meta_json")
    meta = _parse_meta(meta_json)

    return {
        "run_id": r.get("run_id"),
        "symbol": r.get("symbol"),
        "name": r.get("name"),
        "market": r.get("market"),
        "score": score,
        # ✅ 프론트에서 쓰기 쉬운 키
        "meta": meta,
        # (호환/디버그용) 원본도 같이 내려줌
        "meta_json": meta_json,
    }


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

    items = [_norm_row(dict(r)) for r in rows]
    return {"ok": True, "run_id": run_id, "items": items}


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

    items = [_norm_row(dict(r)) for r in rows]
    return {
        "ok": True,
        "run_id": run["id"],
        "asof_date": run["asof_date"],
        "market": run["market"],
        "items": items,
    }