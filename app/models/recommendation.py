# app/models/recommendation.py
from __future__ import annotations

from sqlalchemy import BigInteger, Integer, String, Date, DateTime, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    run_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("screen_runs.id", ondelete="CASCADE"), index=True
    )

    symbol: Mapped[str] = mapped_column(String(16), index=True)
    name: Mapped[str | None] = mapped_column(String(64), nullable=True)

    score: Mapped[float] = mapped_column(Integer, default=0)  # 필요하면 Float로 바꿔도 됨
    rank: Mapped[int] = mapped_column(Integer, default=0)

    asof_date: Mapped[Date | None] = mapped_column(Date, nullable=True)

    created_at: Mapped[DateTime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )