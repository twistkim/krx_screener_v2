from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("")
async def health():
    return {"ok": True}


@router.get("/db")
async def health_db(db: AsyncSession = Depends(get_db)):
    row = (await db.execute(text("SELECT 1 AS ok"))).mappings().first()
    return {"ok": bool(row and row["ok"] == 1)}