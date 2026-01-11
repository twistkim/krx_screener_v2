from __future__ import annotations

from sqlalchemy import BigInteger, Float, Integer, String, Index
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class ScreenPick(Base):
    __tablename__ = "screen_picks"
    __table_args__ = (
        Index("ix_screen_picks_run_rank", "run_id", "rank"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(BigInteger, index=True)

    ticker: Mapped[str] = mapped_column(String(16), index=True)
    score: Mapped[float] = mapped_column(Float)
    rank: Mapped[int] = mapped_column(Integer)
    note: Mapped[str | None] = mapped_column(String(255), nullable=True)