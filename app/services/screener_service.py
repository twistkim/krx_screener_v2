# app/services/screener_service.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


# --- screening thresholds (tune here) ---
LIQUID_MIN_VALUE_20 = 500_000_000      # 최근 20일 평균 거래대금(원)
RANGE30_MAX = 0.22                    # 최근 30일 고저폭/종가 (횡보 판정) — 완화
VOL_RATIO_MIN = 1.1                   # 최근 거래량/20일 평균 거래량 — 완화
BREAKOUT_TOL = 0.98                   # 직전 20일 종가 최고 대비 (2% 아래까지 허용) — 완화

# --- NEW: "고점→하락→(60/120 아래) 횡보→바닥 수렴" 컨텍스트 ---
# 60/120MA 컨텍스트가 필요한 최소 데이터
MIN_BARS_FOR_TREND = 140              # 120MA + slope 계산 여유

# 최근 30일 종가 중 60/120 아래에 머무른 비율 (횡보 구간이 "아래에서" 형성되었는지)
BELOW_LONG_MA_PCT = 0.55              # 55% 이상이면 "장기이평 아래 횡보"로 인정 — 완화

# 바닥 수렴(압축): 최근 10일 변동폭이 최근 30일 변동폭 대비 충분히 줄었는지
COMPRESSION_RATIO = 0.85              # range_10 <= range_30 * 0.85 — 완화

# "우상향→우하향(꺾임)" 컨텍스트: 60MA 기울기가 꺾여 내려오는지(최근 vs 과거)
SLOPE_SHIFT = 10                      # 10거래일 shift로 60MA 기울기 판단

# 아직 크게 오른 종목 제외(장기이평 재탈환/급등) - "바닥권"만 남기기 위한 추가 컷
MA60_RECLAIM_MAX = 1.08               # 종가가 60MA의 +8%를 넘으면 이미 올라탄 걸로 봄 — 완화
MA120_RECLAIM_MAX = 1.08              # 종가가 120MA의 +8%를 넘으면 이미 올라탄 걸로 봄 — 완화

# Overheat ("연간 급등" 제외) — 데이터가 있을 때만 적용
MAX_RET_LOOKBACK = 1.0                # 1년(또는 대체 lookback) 수익률 +100% 초과면 제외
MAX_MA200_GAP = 0.5                   # 200MA 대비 +50% 초과면 제외
NEAR_HIGH_PCT = 0.985                 # 52주 고점 1.5% 이내면 near-high
NEAR_HIGH_RET_CAP = 0.6               # near-high 이면서 수익률 +60% 초과면 제외


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


def _lin_slope(y: list[float]) -> float | None:
    """Return slope of y over x=0..n-1 using least squares. y is chronological (oldest->newest)."""
    n = len(y)
    if n < 2:
        return None
    xs = list(range(n))
    x_mean = (n - 1) / 2.0
    y_mean = _avg(y)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return None
    numer = sum((x - x_mean) * (yy - y_mean) for x, yy in zip(xs, y))
    return numer / denom


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

    # --- NEW: "좌상단 → 우하향"(최근 하락) 시각 패턴을 최우선 신호로 점수화 ---
    # 최근 20일이 (고점 대비 내려오고 + 추세 기울기/수익률이 음수)면 강한 가점
    recent20_closes_latest_first = [float(b.close) for b in bars60[:20]]  # 최신->과거
    recent20_closes = list(reversed(recent20_closes_latest_first))        # 과거->최신

    ret_20 = None
    if recent20_closes and recent20_closes[0] > 0:
        ret_20 = (last_close / recent20_closes[0]) - 1.0

    high20 = _max(recent20_closes)
    drop_from_high_20 = None
    if high20 and high20 > 0:
        drop_from_high_20 = (high20 - last_close) / high20  # 0~1

    slope20 = _lin_slope(recent20_closes)  # 원 단위/일
    slope20_pct_per_day = None
    if slope20 is not None and last_close > 0:
        slope20_pct_per_day = (slope20 / last_close) * 100.0

    downtrend_20 = False
    if (ret_20 is not None) and (slope20 is not None):
        downtrend_20 = (ret_20 < 0) and (slope20 < 0)

    # --- NEW: 주봉(Proxy) 우하향 보조 점수 ---
    # 일봉 데이터를 5거래일 단위로 샘플링해서 주봉 흐름을 대략적으로 추정
    weekly_closes_latest_first = [float(b.close) for b in bars_desc[:60:5]]  # 최신->과거, 약 12주
    weekly_closes = list(reversed(weekly_closes_latest_first))               # 과거->최신

    ret_w = None
    if len(weekly_closes) >= 2 and weekly_closes[0] > 0:
        ret_w = (weekly_closes[-1] / weekly_closes[0]) - 1.0

    slope_w = _lin_slope(weekly_closes)  # 원 단위/주(샘플 기준)
    slope_w_pct_per_week = None
    if slope_w is not None and weekly_closes and weekly_closes[-1] > 0:
        slope_w_pct_per_week = (slope_w / weekly_closes[-1]) * 100.0

    downtrend_w = False
    if (ret_w is not None) and (slope_w is not None):
        downtrend_w = (ret_w < 0) and (slope_w < 0)

    # --- context features: 하락(일봉) / 아래 횡보 / 수렴 / (선택) 돌파 ---
    n_available = len(bars_desc)

    # defaults
    ma60 = None
    ma120 = None
    ma60_prev = None
    slope_down = False
    ma_stack_bearish = False
    downtrend_context_ok = False

    below_ratio = None
    below_long_ma_ok = False

    range_10 = None
    compression_ok = False

    reclaimed_above_ma60 = None
    reclaimed_above_ma120 = None

    if n_available >= MIN_BARS_FOR_TREND:
        closes_all = [float(b.close) for b in bars_desc]  # 최신->과거

        ma60 = _ma(closes_all, 60, 0)
        ma120 = _ma(closes_all, 120, 0)
        ma60_prev = _ma(closes_all, 60, SLOPE_SHIFT)

        # 1) "우상향→우하향(꺾임)"을 가장 중요 신호로 사용
        if ma60 is not None and ma60_prev is not None:
            slope_down = ma60 < ma60_prev
        if ma60 is not None and ma120 is not None:
            ma_stack_bearish = ma60 < ma120

        # downtrend_context_ok는 *점수 가점용* (하드 필터로는 쓰지 않음)
        downtrend_context_ok = slope_down and ma_stack_bearish

        # 2) "60/120 아래 횡보" 비율 (가점/참고)
        if ma60 is not None and ma120 is not None:
            recent30_closes = [float(b.close) for b in bars_desc[:30]]
            below_cnt = 0
            for c in recent30_closes:
                if (c < ma60) and (c < ma120):
                    below_cnt += 1
            below_ratio = (below_cnt / len(recent30_closes)) if recent30_closes else 0.0
            below_long_ma_ok = below_ratio >= BELOW_LONG_MA_PCT

        # 3) "바닥권 수렴" (완화: 10일 range가 30일 대비 충분히 줄면 가점)
        highs10 = [float(b.high) for b in bars_desc[:10]]
        lows10 = [float(b.low) for b in bars_desc[:10]]
        if highs10 and lows10:
            range_10 = (_max(highs10) - _min(lows10)) / last_close
            compression_ok = range_10 <= (range_30 * COMPRESSION_RATIO)

        # 4) 장기이평 "재탈환" 여부 (이제는 탈락이 아니라 감점/정보)
        if ma60 is not None:
            reclaimed_above_ma60 = last_close > ma60 * MA60_RECLAIM_MAX
        if ma120 is not None:
            reclaimed_above_ma120 = last_close > ma120 * MA120_RECLAIM_MAX

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

    # 최종 필터(완화): 유동성/횡보/거래량스파이크/과열제외만 하드 필터로 유지
    if not (liquid_ok and sideways_ok and volume_spike_ok and overheated_ok):
        return None

    # 점수(휴리스틱, 우선순위):
    # 1) "좌상단→우하향"(최근 20일 하락) 시각 패턴을 가장 크게 반영
    # 2) 그 다음으로 60/120 기반 하락 전환 컨텍스트(보조)
    # 3) 거래량비(수급) / 박스권(변동성 패널티) / 아래횡보/수렴/돌파는 보조
    ma_gap_pct = (last_close / ma20 - 1.0) * 100.0

    score = 0.0

    # (A) "좌상단→우하향"(최근 20일 하락) 최우선 가점
    # - 하락 추세 자체가 명확하면 큰 가점
    if downtrend_20:
        score += 35.0

    # - 최근 20일 고점에서 얼마나 내려왔는지(좌상단에서 내려온 느낌) 추가 가점
    if drop_from_high_20 is not None:
        # 10% 하락이면 +10, 30%면 +20 수준으로 제한
        score += min(drop_from_high_20 * 100.0, 22.0)

    # - 20일 기울기가 더 가팔라질수록 추가 가점(퍼센트/일 기준)
    if slope20_pct_per_day is not None and slope20_pct_per_day < 0:
        score += min((-slope20_pct_per_day) * 6.0, 18.0)

    # (A-2) 주봉(Proxy) 우하향 보조 가점 (일봉보다 작게)
    if downtrend_w:
        score += 8.0
    if slope_w_pct_per_week is not None and slope_w_pct_per_week < 0:
        score += min((-slope_w_pct_per_week) * 2.5, 10.0)

    # (B) 60/120 기반 하락 전환 컨텍스트(보조)
    if slope_down:
        score += 10.0
    if ma_stack_bearish:
        score += 8.0
    if downtrend_context_ok:
        score += 6.0

    # (C) 수급(거래량 스파이크) 가점
    score += (vol_ratio * 6.0)

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
        "ret_20": float(ret_20) if ret_20 is not None else None,
        "drop_from_high_20": float(drop_from_high_20) if drop_from_high_20 is not None else None,
        "slope20": float(slope20) if slope20 is not None else None,
        "slope20_pct_per_day": float(slope20_pct_per_day) if slope20_pct_per_day is not None else None,
        "ret_w": float(ret_w) if ret_w is not None else None,
        "slope_w": float(slope_w) if slope_w is not None else None,
        "slope_w_pct_per_week": float(slope_w_pct_per_week) if slope_w_pct_per_week is not None else None,
        "filters": {
            "liquid_ok": liquid_ok,
            "sideways_ok": sideways_ok,
            "volume_spike_ok": volume_spike_ok,
            "breakout_ok": breakout_ok,
            "overheated_ok": overheated_ok,
            "downtrend_context_ok": downtrend_context_ok,
            "slope_down": slope_down,
            "ma_stack_bearish": ma_stack_bearish,
            "below_long_ma_ok": below_long_ma_ok,
            "compression_ok": compression_ok,
            "reclaimed_above_ma60": bool(reclaimed_above_ma60) if reclaimed_above_ma60 is not None else None,
            "reclaimed_above_ma120": bool(reclaimed_above_ma120) if reclaimed_above_ma120 is not None else None,
            "downtrend_20": bool(downtrend_20),
            "downtrend_w": bool(downtrend_w),
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
