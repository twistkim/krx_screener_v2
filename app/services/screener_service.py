# app/services/screener_service.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


# --- screening thresholds (tune here) ---
LIQUID_MIN_VALUE_20 = 500_000_000      # 최근 20일 평균 거래대금(원)
RANGE30_MAX = 0.18                    # 최근 30일 고저폭/종가 (횡보 판정)
VOL_RATIO_MIN = 1.2                   # 최근 거래량/20일 평균 거래량
BREAKOUT_TOL = 0.995                  # 직전 20일 종가 최고 대비 (0.5% 아래까지 허용)

# --- NEW: "고점→하락→(60/120 아래) 횡보→바닥 수렴" 컨텍스트 ---
# 60/120MA 컨텍스트가 필요한 최소 데이터
MIN_BARS_FOR_TREND = 140              # 120MA + slope 계산 여유

# 최근 30일 종가 중 60/120 아래에 머무른 비율 (횡보 구간이 "아래에서" 형성되었는지)
BELOW_LONG_MA_PCT = 0.7               # 70% 이상이면 "장기이평 아래 횡보"로 인정

# 바닥 수렴(압축): 최근 10일 변동폭이 최근 30일 변동폭 대비 충분히 줄었는지
COMPRESSION_RATIO = 0.65              # range_10 <= range_30 * 0.65

# "우상향→우하향(꺾임)" 컨텍스트: 60MA 기울기가 꺾여 내려오는지(최근 vs 과거)
SLOPE_SHIFT = 10                      # 10거래일 shift로 60MA 기울기 판단

# 아직 크게 오른 종목 제외(장기이평 재탈환/급등) - "바닥권"만 남기기 위한 추가 컷
MA60_RECLAIM_MAX = 1.02               # 종가가 60MA의 +2%를 넘으면 이미 올라탄 걸로 보고 제외(보수적)
MA120_RECLAIM_MAX = 1.02              # 종가가 120MA의 +2%를 넘으면 제외

# Overheat ("연간 급등" 제외) — 데이터가 있을 때만 적용
MAX_RET_LOOKBACK = 1.5                # lookback 기간 수익률 +150% 초과면 제외
MAX_MA200_GAP = 0.6                   # 200MA 대비 +60% 초과면 제외
NEAR_HIGH_PCT = 0.98                  # 52주(또는 가용 구간) 고점 2% 이내면 near-high
NEAR_HIGH_RET_CAP = 0.8               # near-high 이면서 수익률 +80% 초과면 제외


@dataclass
class Bar:
    trade_date: str
    open: int
    high: int
    low: int
    close: int
    volume: int
    value: int


def _avg(nums: list[float]) -> float:
    return sum(nums) / len(nums) if nums else 0.0


def _max(nums: list[float]) -> float:
    return max(nums) if nums else 0.0


def _min(nums: list[float]) -> float:
    return min(nums) if nums else 0.0


def _ma(series: list[float], window: int, start: int = 0) -> float | None:
    """series는 '최신이 앞'인 리스트를 가정. start부터 window개 평균."""
    end = start + window
    if len(series) < end:
        return None
    chunk = series[start:end]
    return _avg(chunk) if chunk else None


async def _get_latest_trade_date(db: AsyncSession, market: str) -> str | None:
    if market == "ALL":
        q = text("SELECT MAX(trade_date) AS d FROM daily_bars")
        row = (await db.execute(q)).mappings().first()
    else:
        q = text(
            """
            SELECT MAX(b.trade_date) AS d
            FROM daily_bars b
            JOIN symbols s ON s.symbol = b.symbol
            WHERE s.market=:m
            """
        )
        row = (await db.execute(q, {"m": market})).mappings().first()
    return row["d"] if row and row["d"] else None


async def _get_candidate_symbols(db: AsyncSession, market: str, asof_date: str, limit: int) -> list[str]:
    """
    asof_date 당일 거래대금(value) 상위 종목을 후보로 선정 (쿼리 1번으로 후보군 축소)
    """
    if market == "ALL":
        q = text(
            """
            SELECT symbol
            FROM daily_bars
            WHERE trade_date=:d
            ORDER BY value DESC
            LIMIT :lim
            """
        )
        rows = (await db.execute(q, {"d": asof_date, "lim": limit})).mappings().all()
    else:
        q = text(
            """
            SELECT b.symbol
            FROM daily_bars b
            JOIN symbols s ON s.symbol = b.symbol
            WHERE b.trade_date=:d AND s.market=:m
            ORDER BY b.value DESC
            LIMIT :lim
            """
        )
        rows = (await db.execute(q, {"d": asof_date, "m": market, "lim": limit})).mappings().all()

    return [r["symbol"] for r in rows]


async def _get_recent_bars(db: AsyncSession, symbol: str, limit: int = 260) -> list[Bar]:
    q = text(
        """
        SELECT trade_date, open, high, low, close, volume, value
        FROM daily_bars
        WHERE symbol=:s
        ORDER BY trade_date DESC
        LIMIT :lim
        """
    )
    rows = (await db.execute(q, {"s": symbol, "lim": limit})).mappings().all()
    bars = [
        Bar(
            trade_date=r["trade_date"],
            open=int(r["open"] or 0),
            high=int(r["high"] or 0),
            low=int(r["low"] or 0),
            close=int(r["close"] or 0),
            volume=int(r["volume"] or 0),
            value=int(r["value"] or 0),
        )
        for r in rows
    ]
    return bars


def _score_symbol(bars_desc: list[Bar]) -> tuple[float, dict[str, Any]] | None:
    """
    휴리스틱 스크리닝
    - bars_desc: 최신 -> 과거
    """
    if len(bars_desc) < 60:
        return None

    # 최신 60봉 기준 (최신이 앞)
    bars60 = bars_desc[:60]
    closes60 = [float(b.close) for b in bars60]
    highs30 = [float(b.high) for b in bars60[:30]]
    lows30 = [float(b.low) for b in bars60[:30]]
    vols20 = [float(b.volume) for b in bars60[:20]]
    vals20 = [float(b.value) for b in bars60[:20]]

    last = bars60[0]
    last_close = float(last.close)
    if last_close <= 0:
        return None

    ma20 = _avg(closes60[:20])
    if ma20 <= 0:
        return None

    # 유동성: 최근 20일 평균 거래대금
    avg_value_20 = _avg(vals20)  # KRW
    liquid_ok = avg_value_20 >= LIQUID_MIN_VALUE_20

    # 횡보(단순): 최근 30일 고저폭이 너무 크면 제외
    range_30 = (_max(highs30) - _min(lows30)) / last_close
    sideways_ok = range_30 <= RANGE30_MAX

    # 거래량 스파이크(단순)
    avg_vol_20 = _avg(vols20)
    vol_ratio = (float(last.volume) / avg_vol_20) if avg_vol_20 > 0 else 0.0
    volume_spike_ok = vol_ratio >= VOL_RATIO_MIN

    # --- NEW: 바닥권/하락 컨텍스트 (60/120 아래 횡보 + 수렴 + "꺾임") ---
    n_available = len(bars_desc)
    downtrend_context_ok = True
    below_long_ma_ok = True
    compression_ok = True
    not_reclaimed_long_ma_ok = True

    ma60 = None
    ma120 = None
    ma60_prev = None

    if n_available >= MIN_BARS_FOR_TREND:
        closes_all = [float(b.close) for b in bars_desc]  # 최신->과거

        ma60 = _ma(closes_all, 60, 0)
        ma120 = _ma(closes_all, 120, 0)

        # 60MA 기울기(최근 vs 10일 전)
        ma60_prev = _ma(closes_all, 60, SLOPE_SHIFT)

        # 1) "우상향→우하향" 컨텍스트: 60MA가 꺾여 내려오는지 + 60<120(약세 구간) 선호
        if ma60 is not None and ma60_prev is not None and ma120 is not None:
            slope_down = ma60 < ma60_prev
            ma_stack_bearish = ma60 < ma120
            downtrend_context_ok = slope_down and ma_stack_bearish

        # 2) "60/120 아래 횡보": 최근 30일 종가 중 장기이평 아래 비율
        if ma60 is not None and ma120 is not None:
            recent30_closes = [float(b.close) for b in bars_desc[:30]]
            below_cnt = 0
            for c in recent30_closes:
                if (c < ma60) and (c < ma120):
                    below_cnt += 1
            below_ratio = (below_cnt / len(recent30_closes)) if recent30_closes else 0.0
            below_long_ma_ok = below_ratio >= BELOW_LONG_MA_PCT
        else:
            below_ratio = None

        # 3) "바닥권 수렴": 10일 변동폭이 30일 변동폭 대비 압축되는지
        highs10 = [float(b.high) for b in bars_desc[:10]]
        lows10 = [float(b.low) for b in bars_desc[:10]]
        range_10 = ((_max(highs10) - _min(lows10)) / last_close) if highs10 and lows10 else None
        compression_ok = (range_10 is not None) and (range_10 <= (range_30 * COMPRESSION_RATIO))

        # 4) "상승한 종목 제외": 이미 60/120 위로 확 올라탄 건 제외 (바닥권만)
        if ma60 is not None and ma120 is not None:
            not_reclaimed_long_ma_ok = (last_close <= ma60 * MA60_RECLAIM_MAX) and (last_close <= ma120 * MA120_RECLAIM_MAX)
        else:
            not_reclaimed_long_ma_ok = True
    else:
        below_ratio = None
        range_10 = None

    # --- breakout: "횡보+거래 터짐 이후 위로 방향" (기존보다 약간 현실적으로) ---
    # 기존: prev20_max_close 돌파 + last_close >= ma20
    # 그대로 두되, 너무 빡세면 BREAKOUT_TOL/조건만 튜닝하면 됨.
    prev20_max_close = _max([float(b.close) for b in bars60[1:21]])
    breakout_ok = (last_close >= (prev20_max_close * BREAKOUT_TOL)) and (last_close >= ma20)

    # --- overheat filter: "이미 너무 오른 종목" 제외 (데이터가 있을 때만 적용) ---
    # lookback 수익률: 가능하면 252(1년) 사용, 아니면 120(반년 정도)로 대체
    lookback_n = 252 if n_available >= 253 else (120 if n_available >= 121 else None)
    ret_lookback = None
    if lookback_n is not None:
        past_close = float(bars_desc[lookback_n].close)
        if past_close > 0:
            ret_lookback = (last_close / past_close) - 1.0

    # 200일선 과열: 200일선이 있을 때만
    ma200 = None
    ma200_gap = None
    if n_available >= 200:
        closes200 = [float(b.close) for b in bars_desc[:200]]
        ma200 = _avg(closes200)
        if ma200 and ma200 > 0:
            ma200_gap = (last_close / ma200) - 1.0

    # 52주(또는 가용 구간) 고점 근접 여부
    high_window = 252 if n_available >= 252 else n_available
    highs_window = [float(b.high) for b in bars_desc[:high_window]]
    high_lookback = _max(highs_window)
    near_high = (last_close >= high_lookback * NEAR_HIGH_PCT) if high_lookback > 0 else False

    overheated_ok = True
    if ret_lookback is not None and ret_lookback > MAX_RET_LOOKBACK:
        overheated_ok = False
    if ma200_gap is not None and ma200_gap > MAX_MA200_GAP:
        overheated_ok = False
    if near_high and (ret_lookback is not None) and (ret_lookback > NEAR_HIGH_RET_CAP):
        overheated_ok = False

    # 최종 필터: 기존 4개 + overheat + NEW 컨텍스트
    if not (
        liquid_ok
        and sideways_ok
        and volume_spike_ok
        and breakout_ok
        and overheated_ok
        and downtrend_context_ok
        and below_long_ma_ok
        and compression_ok
        and not_reclaimed_long_ma_ok
    ):
        return None

    # 점수(휴리스틱): MA 대비 상승률 + 거래량비 - 변동성 패널티
    ma_gap_pct = (last_close / ma20 - 1.0) * 100.0
    score = ma_gap_pct + (vol_ratio * 5.0) - (range_30 * 50.0)

    meta = {
        "last_close": int(last_close),
        "ma20": float(ma20),
        "ma60": float(ma60) if ma60 is not None else None,
        "ma120": float(ma120) if ma120 is not None else None,
        "ma60_prev": float(ma60_prev) if ma60_prev is not None else None,
        "ma_gap_pct": float(ma_gap_pct),
        "avg_value_20": float(avg_value_20),
        "range_30": float(range_30),
        "range_10": float(range_10) if range_10 is not None else None,
        "below_long_ma_ratio_30": float(below_ratio) if below_ratio is not None else None,
        "vol_ratio": float(vol_ratio),
        "ret_lookback": float(ret_lookback) if ret_lookback is not None else None,
        "ma200": float(ma200) if ma200 is not None else None,
        "ma200_gap_pct": float(ma200_gap * 100.0) if ma200_gap is not None else None,
        "high_lookback": float(high_lookback) if high_lookback is not None else None,
        "near_high": bool(near_high),
        "filters": {
            "liquid_ok": liquid_ok,
            "sideways_ok": sideways_ok,
            "volume_spike_ok": volume_spike_ok,
            "breakout_ok": breakout_ok,
            "overheated_ok": overheated_ok,
            "downtrend_context_ok": downtrend_context_ok,
            "below_long_ma_ok": below_long_ma_ok,
            "compression_ok": compression_ok,
            "not_reclaimed_long_ma_ok": not_reclaimed_long_ma_ok,
        },
    }
    return score, meta


async def _insert_screen_run(db: AsyncSession, asof_date: str, market: str) -> int:
    q = text("INSERT INTO screen_runs (asof_date, market) VALUES (:d, :m)")
    await db.execute(q, {"d": asof_date, "m": market})
    row = (await db.execute(text("SELECT LAST_INSERT_ID() AS id"))).mappings().first()
    return int(row["id"])


async def _insert_recos(db: AsyncSession, run_id: int, recos: list[dict[str, Any]]) -> int:
    if not recos:
        return 0

    q = text(
        """
        INSERT INTO recommendations (run_id, symbol, `rank`, score, meta_json)
        VALUES (:run_id, :symbol, :rank, :score, :meta_json)
        """
    )
    params = []
    for i, r in enumerate(recos, start=1):
        params.append(
            {
                "run_id": run_id,
                "symbol": r["symbol"],
                "rank": i,
                "score": r["score"],
                "meta_json": r["meta_json"],
            }
        )
    await db.execute(q, params)
    return len(params)


async def run_screen(db: AsyncSession, market: str = "ALL", top_n: int = 50) -> dict[str, Any]:
    asof_date = await _get_latest_trade_date(db, market=market)
    if not asof_date:
        return {
            "ok": True,
            "run_id": None,
            "asof_date": None,
            "market": market,
            "candidates": 0,
            "saved": 0,
            "top": [],
            "stats": {"reason": "no daily_bars data"},
        }

    # 후보군 축소(성능 핵심): 당일 거래대금 상위 N개만
    candidate_limit = max(800, top_n * 10)
    symbols = await _get_candidate_symbols(db, market=market, asof_date=asof_date, limit=candidate_limit)

    enough_data = 0
    passed = 0
    selected: list[dict[str, Any]] = []

    for sym in symbols:
        bars = await _get_recent_bars(db, sym, limit=260)
        if len(bars) < 60:
            continue
        enough_data += 1

        scored = _score_symbol(bars)
        if not scored:
            continue
        score, meta = scored
        passed += 1
        selected.append(
            {
                "symbol": sym,
                "score": float(score),
                "meta_json": json.dumps(meta, ensure_ascii=False),
                "meta": meta,
            }
        )

    selected.sort(key=lambda x: x["score"], reverse=True)
    top = selected[:top_n]

    # 저장
    run_id = await _insert_screen_run(db, asof_date=asof_date, market=market)
    saved = await _insert_recos(db, run_id=run_id, recos=top)
    await db.commit()

    return {
        "ok": True,
        "run_id": run_id,
        "asof_date": asof_date,
        "market": market,
        "candidates": len(symbols),
        "saved": saved,
        "top": [{"symbol": r["symbol"], "score": r["score"], "meta": r["meta"]} for r in top],
        "stats": {
            "symbols_candidates": len(symbols),
            "enough_data": enough_data,
            "passed_filters": passed,
            "final": len(top),
        },
    }
