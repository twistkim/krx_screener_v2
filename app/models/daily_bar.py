from __future__ import annotations

from sqlalchemy import BigInteger, Date, Float, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class DailyBar(Base):
    __tablename__ = "daily_bars"
    __table_args__ = (
        UniqueConstraint("ticker", "trade_date", name="uq_daily_bars_ticker_date"),
        Index("ix_daily_bars_trade_date", "trade_date"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    ticker: Mapped[str] = mapped_column(String(16), index=True)
    market: Mapped[str] = mapped_column(String(16), index=True)
    trade_date: Mapped[object] = mapped_column(Date)

    open: Mapped[int] = mapped_column(Integer)
    high: Mapped[int] = mapped_column(Integer)
    low: Mapped[int] = mapped_column(Integer)
    close: Mapped[int] = mapped_column(Integer)
    volume: Mapped[int] = mapped_column(BigInteger)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)