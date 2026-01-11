from __future__ import annotations

from sqlalchemy import Date, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class Symbol(Base):
    __tablename__ = "symbols"

    ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    market: Mapped[str] = mapped_column(String(16), index=True)
    name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_seen: Mapped[object | None] = mapped_column(Date, nullable=True)  # date