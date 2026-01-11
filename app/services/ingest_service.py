# app/services/ingest_service.py
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, date
from typing import Any

import requests

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

import pandas as pd
import FinanceDataReader as fdr

# --- Safety: prevent hangs in FinanceDataReader/requests ---
# FinanceDataReader uses `requests` internally; without a timeout, a single stalled HTTP call
# can hang the whole ingest request indefinitely.
_old_request = requests.Session.request


def _request_with_timeout(self, method, url, **kwargs):
    kwargs.setdefault("timeout", 20)  # seconds
    return _old_request(self, method, url, **kwargs)


requests.Session.request = _request_with_timeout
# ----------------------------------------------------------


def _to_yyyymmdd(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def _parse_yyyymmdd(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d")


def _parse_any_date(s: str) -> datetime:
    """Parse either 'yyyymmdd' or 'yyyy-mm-dd' (also accepts 'yyyy/mm/dd')."""
    s = (s or "").strip()
    if not s:
        raise ValueError("empty date")
    if "-" in s:
        return datetime.strptime(s, "%Y-%m-%d")
    if "/" in s:
        return datetime.strptime(s, "%Y/%m/%d")
    # fallback: yyyymmdd
    return _parse_yyyymmdd(s)


async def _count_symbols(db: AsyncSession, market: str) -> int:
    if market == "ALL":
        q = text("SELECT COUNT(*) AS c FROM symbols")
        row = (await db.execute(q)).mappings().first()
    else:
        q = text("SELECT COUNT(*) AS c FROM symbols WHERE market=:m")
        row = (await db.execute(q, {"m": market})).mappings().first()
    return int(row["c"] or 0) if row else 0


async def _count_daily_rows_for_date(db: AsyncSession, trade_date: str, market: str) -> int:
    """
    trade_date: yyyymmdd
    """
    if market == "ALL":
        q = text("SELECT COUNT(*) AS c FROM daily_bars WHERE trade_date=:d")
        row = (await db.execute(q, {"d": trade_date})).mappings().first()
    else:
        q = text(
            """
            SELECT COUNT(*) AS c
            FROM daily_bars b
            JOIN symbols s ON s.symbol = b.symbol
            WHERE b.trade_date=:d AND s.market=:m
            """
        )
        row = (await db.execute(q, {"d": trade_date, "m": market})).mappings().first()
    return int(row["c"] or 0) if row else 0


async def _upsert_symbols_minimal(db: AsyncSession, tickers: list[str], market: str) -> int:
    """
    최소 컬럼만 upsert (symbol, market, name(optional))
    - name까지 채우면 매우 느려질 수 있어서, 기본은 NULL로 둠.
    """
    if not tickers:
        return 0

    sql = text(
        """
        INSERT INTO symbols (symbol, market, name)
        VALUES (:symbol, :market, :name) AS new
        ON DUPLICATE KEY UPDATE
            market = new.market,
            name = COALESCE(new.name, symbols.name)
        """
    )

    params = [{"symbol": t, "market": market, "name": None} for t in tickers]
    await db.execute(sql, params)
    return len(params)


async def _upsert_daily_bars(db: AsyncSession, trade_date: str, rows: list[dict[str, Any]]) -> int:
    """
    rows: [{symbol, open, high, low, close, volume, value}, ...]
    daily_bars는 (symbol, trade_date) UNIQUE/PK가 있어야 ON DUPLICATE KEY UPDATE가 먹음.
    """
    if not rows:
        return 0

    sql = text(
        """
        INSERT INTO daily_bars
            (symbol, trade_date, open, high, low, close, volume, value)
        VALUES
            (:symbol, :trade_date, :open, :high, :low, :close, :volume, :value) AS new
        ON DUPLICATE KEY UPDATE
            open = new.open,
            high = new.high,
            low = new.low,
            close = new.close,
            volume = new.volume,
            value = new.value
        """
    )

    # executemany
    params = []
    for r in rows:
        params.append(
            {
                "symbol": r["symbol"],
                "trade_date": trade_date,
                "open": r["open"],
                "high": r["high"],
                "low": r["low"],
                "close": r["close"],
                "volume": r["volume"],
                "value": r["value"],
            }
        )

    await db.execute(sql, params)
    return len(params)


async def _upsert_daily_bars_rows(db: AsyncSession, rows: list[dict[str, Any]]) -> int:
    """rows: [{symbol, trade_date(yyyymmdd), open, high, low, close, volume, value}, ...]"""
    if not rows:
        return 0

    sql = text(
        """
        INSERT INTO daily_bars
            (symbol, trade_date, open, high, low, close, volume, value)
        VALUES
            (:symbol, :trade_date, :open, :high, :low, :close, :volume, :value) AS new
        ON DUPLICATE KEY UPDATE
            open = new.open,
            high = new.high,
            low = new.low,
            close = new.close,
            volume = new.volume,
            value = new.value
        """
    )

    await db.execute(sql, rows)
    return len(rows)


def _to_yyyy_mm_dd(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def _load_krx_listing() -> pd.DataFrame:
    """KRX 전체 상장 목록(현재 기준)."""
    df = fdr.StockListing("KRX")
    # 표준화: Code/Name/Market 컬럼이 있다고 가정
    # FinanceDataReader 버전에 따라 컬럼명이 다를 수 있어 방어적으로 처리
    rename_map = {}
    for c in df.columns:
        if c.lower() == "code":
            rename_map[c] = "Code"
        elif c.lower() == "name":
            rename_map[c] = "Name"
        elif c.lower() == "market":
            rename_map[c] = "Market"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _filter_listing_by_market(df: pd.DataFrame, market: str) -> pd.DataFrame:
    if market == "ALL":
        return df
    if "Market" not in df.columns:
        return df
    return df[df["Market"].astype(str).str.upper() == market.upper()]


def _pick_reference_ticker(codes: list[str]) -> str:
    # 거래일 캘린더 추출용 기준 종목(삼성전자 우선)
    return "005930" if "005930" in codes else (codes[0] if codes else "005930")


def _fetch_prices_df(code: str, start: str, end: str) -> pd.DataFrame:
    """FinanceDataReader DataReader는 blocking 이므로 thread에서 호출될 수 있음.

    NOTE:
    - FDR 내부에서 Yahoo 등으로 라우팅되며 404/네트워크 오류가 발생할 수 있음.
    - ingest 전체가 500으로 죽지 않도록 여기서 예외를 흡수하고 빈 DF를 반환한다.
    """
    try:
        df = fdr.DataReader(code, start, end)
    except Exception:
        return pd.DataFrame()

    if df is None:
        return pd.DataFrame()

    # 컬럼 표준화(대부분 Open/High/Low/Close/Volume)
    rename_map = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc == "open":
            rename_map[c] = "Open"
        elif lc == "high":
            rename_map[c] = "High"
        elif lc == "low":
            rename_map[c] = "Low"
        elif lc == "close":
            rename_map[c] = "Close"
        elif lc == "volume":
            rename_map[c] = "Volume"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


async def _get_existing_trade_dates(db: AsyncSession, market: str, start_yyyymmdd: str, end_yyyymmdd: str) -> set[str]:
    """DB에 이미 존재하는 trade_date(yyyymmdd) 집합."""
    if market == "ALL":
        q = text(
            """
            SELECT DISTINCT trade_date
            FROM daily_bars
            WHERE trade_date BETWEEN :s AND :e
            """
        )
        rows = (await db.execute(q, {"s": start_yyyymmdd, "e": end_yyyymmdd})).mappings().all()
    else:
        q = text(
            """
            SELECT DISTINCT b.trade_date
            FROM daily_bars b
            JOIN symbols s ON s.symbol = b.symbol
            WHERE s.market = :m
              AND b.trade_date BETWEEN :s AND :e
            """
        )
        rows = (await db.execute(q, {"m": market, "s": start_yyyymmdd, "e": end_yyyymmdd})).mappings().all()
    return {str(r["trade_date"]) for r in rows}


async def ingest_missing(
    db: AsyncSession,
    market: str = "ALL",
    days: int = 60,
    asof: str | None = None,  # yyyymmdd
    start: str | None = None,  # 'yyyy-mm-dd' or 'yyyymmdd'
    end: str | None = None,  # 'yyyy-mm-dd' or 'yyyymmdd'
) -> dict[str, Any]:
    """
    DB 기준으로 '없는 거래일'만 가져오기 (FinanceDataReader 기반).

    핵심 아이디어
    - pykrx의 "하루에 전종목" API가 FDR에는 없으므로, 티커별로 기간(start~end)을 가져온 뒤
      DB에 없는 trade_date만 골라서 upsert 한다.
    - 거래일 캘린더는 기준 종목(삼성전자 등) 가격 데이터의 index(날짜)로 산출.
    """
    # 1) 기준 범위 결정
    # - end 우선 > asof(yyyymmdd) > now
    # - start가 주어지면 해당 범위를 사용, 아니면 기존처럼 days 기반으로 최근 N거래일만 ingest
    try:
        end_dt = _parse_any_date(end) if end else (_parse_yyyymmdd(asof) if asof else datetime.now())
    except Exception:
        return {"ok": False, "market": market, "error": f"end/asof 날짜 파싱 실패: end={end!r}, asof={asof!r}"}

    if start:
        try:
            start_dt = _parse_any_date(start)
        except Exception:
            return {"ok": False, "market": market, "error": f"start 날짜 파싱 실패: start={start!r}"}
        if start_dt > end_dt:
            return {"ok": False, "market": market, "error": f"start가 end보다 큽니다: start={start!r}, end={(end or asof)!r}"}
    else:
        start_dt = None

    end_str = _to_yyyy_mm_dd(end_dt)

    # 2) 상장 목록 로드 + 마켓 필터
    listing = _filter_listing_by_market(_load_krx_listing(), market)

    if "Code" not in listing.columns:
        return {
            "ok": False,
            "market": market,
            "error": "FDR StockListing('KRX') 결과에 Code 컬럼이 없습니다.",
        }

    # Code/Name 정리
    listing = listing.copy()
    listing["Code"] = listing["Code"].astype(str).str.strip()
    # 혹시 문자가 섞여 들어오면 숫자만 남김
    listing["Code"] = listing["Code"].str.replace(r"\D", "", regex=True)
    # 정확히 6자리 숫자만 허용 (빈값/잘못된 값이 000000으로 변환되는 것을 방지)
    listing = listing[listing["Code"].str.fullmatch(r"\d{6}", na=False)]
    # 혹시라도 000000 같은 더미값 제거
    listing = listing[listing["Code"] != "000000"]

    if "Name" not in listing.columns:
        listing["Name"] = None

    codes: list[str] = listing["Code"].dropna().astype(str).tolist()
    names_map = {row["Code"]: row.get("Name") for _, row in listing.iterrows()}

    # 3) symbols upsert (가능하면 name도 채움)
    if codes:
        sql = text(
            """
            INSERT INTO symbols (symbol, market, name)
            VALUES (:symbol, :market, :name) AS new
            ON DUPLICATE KEY UPDATE
                market = new.market,
                name = COALESCE(new.name, symbols.name)
            """
        )

        market_map: dict[str, str] = {}
        if "Market" in listing.columns:
            # listing의 Market 컬럼을 그대로 사용 (KOSPI/KOSDAQ/KONEX 등)
            market_map = {str(r["Code"]): str(r["Market"]) for _, r in listing.iterrows()}

        params = []
        for c in codes:
            if market != "ALL":
                mkt = market
            else:
                mkt = market_map.get(c, "ALL")
            params.append({"symbol": c, "market": mkt, "name": names_map.get(c)})

        await db.execute(sql, params)
        await db.commit()

    symbols_total = await _count_symbols(db, market=market)

    # 4) 거래일 캘린더 추출 (기준 종목 1개로 산출)
    # - start/end 범위가 주어지면: 해당 기간의 전체 거래일을 ingest 대상으로 사용
    # - start가 없으면: 기존처럼 최근 days 거래일만 ingest 대상으로 사용
    ref_code = _pick_reference_ticker(codes)

    if start_dt:
        # start가 휴일이면 첫 거래일이 누락될 수 있어 약간 앞에서부터 조회
        ref_start_dt = start_dt - timedelta(days=30)
        ref_start_str = _to_yyyy_mm_dd(ref_start_dt)
        ref_df = await asyncio.to_thread(_fetch_prices_df, ref_code, ref_start_str, end_str)
    else:
        # 거래일 확보를 위해 넉넉히 과거까지 조회 (최근 days 거래일 확보 목적)
        ref_start_dt = end_dt - timedelta(days=days * 6 + 180)
        ref_start_str = _to_yyyy_mm_dd(ref_start_dt)
        ref_df = await asyncio.to_thread(_fetch_prices_df, ref_code, ref_start_str, end_str)

    if ref_df is None or ref_df.empty:
        return {
            "ok": False,
            "market": market,
            "error": f"기준 종목({ref_code})의 가격 데이터를 가져오지 못했습니다. (FDR)",
        }

    ref_dates = [d.strftime("%Y%m%d") for d in pd.to_datetime(ref_df.index).to_pydatetime()]

    if start_dt:
        start_yyyymmdd = start_dt.strftime("%Y%m%d")
        end_yyyymmdd = end_dt.strftime("%Y%m%d")
        target_dates = [d for d in ref_dates if start_yyyymmdd <= d <= end_yyyymmdd]
        # 실제로 데이터가 있는 첫/마지막 거래일로 보정
        if target_dates:
            start_yyyymmdd = target_dates[0]
            end_yyyymmdd = target_dates[-1]
    else:
        # index는 DatetimeIndex가 보통이며, 마지막 days개를 거래일로 사용
        target_dates = ref_dates[-days:] if len(ref_dates) >= days else ref_dates
        if not target_dates:
            return {"ok": False, "market": market, "error": "거래일 캘린더를 만들 수 없습니다."}
        start_yyyymmdd = target_dates[0]
        end_yyyymmdd = target_dates[-1]

    if not target_dates:
        return {"ok": False, "market": market, "error": "지정한 기간에 거래일이 없습니다."}

    start_str = datetime.strptime(start_yyyymmdd, "%Y%m%d").strftime("%Y-%m-%d")
    end_str = datetime.strptime(end_yyyymmdd, "%Y%m%d").strftime("%Y-%m-%d")

    # 5) 이번 실행에서 upsert할 거래일 집합 (기간 전체)
    # 기존 구현은 "해당 trade_date에 어떤 종목이든 1개라도 있으면" 그 날짜를 존재한다고 판단해서
    # 다른 종목들의 결손이 영원히 채워지지 않는 문제가 생길 수 있다.
    # 따라서 이 ingest는 지정한 기간(start~end)의 데이터를 "항상 upsert" 해서 결손을 자동으로 메운다.
    target_set = set(target_dates)

    trading_days_covered = len(target_dates)
    trading_days_fetched = trading_days_covered

    # 6) 티커별로 기간 데이터를 가져와 기간 전체를 upsert (이미 있는 row는 ON DUPLICATE KEY UPDATE로 갱신)
    #    (FDR 호출은 blocking이므로 thread로 돌리고, 동시에 너무 많이 돌리면 느려지거나 차단될 수 있어 제한)
    semaphore = asyncio.Semaphore(6)
    rows_upserted_total = 0
    calendar_days_scanned = (end_dt - datetime.strptime(start_yyyymmdd, "%Y%m%d")).days + 1

    async def fetch_one(code: str) -> list[dict[str, Any]]:
        async with semaphore:
            df = await asyncio.to_thread(_fetch_prices_df, code, start_str, end_str)
        if df is None or df.empty:
            return []

        # df index -> yyyymmdd
        idx = pd.to_datetime(df.index)
        df = df.copy()
        df["trade_date"] = [d.strftime("%Y%m%d") for d in idx.to_pydatetime()]
        # 기간(target_dates) 밖의 데이터는 제외
        df = df[df["trade_date"].isin(target_set)]
        if df.empty:
            return []

        # value는 FDR에 없을 수 있어 close*volume으로 근사
        rows_local: list[dict[str, Any]] = []
        for _, r in df.iterrows():
            o = int(r.get("Open", 0) or 0)
            h = int(r.get("High", 0) or 0)
            l = int(r.get("Low", 0) or 0)
            c = int(r.get("Close", 0) or 0)
            v = int(r.get("Volume", 0) or 0)
            val = int(c * v) if c and v else 0
            rows_local.append(
                {
                    "symbol": str(code),
                    "trade_date": str(r["trade_date"]),
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                    "value": val,
                }
            )
        return rows_local

    # batch upsert (너무 큰 executemany를 피하기 위해 chunk)
    CHUNK = 5000
    pending: list[dict[str, Any]] = []

    # 병렬 fetch → 결과를 순차적으로 DB에 적재
    tasks = [asyncio.create_task(fetch_one(code)) for code in codes]
    for t in asyncio.as_completed(tasks):
        rows = await t
        if not rows:
            continue

        pending.extend(rows)
        if len(pending) >= CHUNK:
            rows_upserted_total += await _upsert_daily_bars_rows(db, pending)
            await db.commit()
            pending.clear()

    if pending:
        rows_upserted_total += await _upsert_daily_bars_rows(db, pending)
        await db.commit()
        pending.clear()

    return {
        "ok": True,
        "market": market,
        "end": _to_yyyymmdd(end_dt),
        "days_requested": days,
        "calendar_days_scanned": int(calendar_days_scanned),
        "symbols_total": symbols_total,
        "trading_days_covered": trading_days_covered,
        "trading_days_fetched": trading_days_fetched,
        "rows_upserted_total": rows_upserted_total,
        "range_requested": {"start": start, "end": end, "asof": asof, "days": days},
        "range": {"start": start_yyyymmdd, "end": end_yyyymmdd},
    }