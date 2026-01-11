// app/static/js/pages/home.js

async function api(url, opts = {}) {
  const res = await fetch(url, {
    // keep JSON default, but allow overriding
    headers: { Accept: "application/json", ...(opts.headers || {}) },
    credentials: opts.credentials || "same-origin",
    ...opts,
  });

  const ct = res.headers.get("content-type") || "";
  const data = ct.includes("application/json") ? await res.json() : await res.text();

  if (!res.ok) {
    const msg = typeof data === "string" ? data : (data.detail || data.message || `HTTP ${res.status}`);
    throw new Error(msg);
  }
  return data;
}

function $(id) {
  return document.getElementById(id);
}

function elExists(id) {
  return !!$(id);
}

function nowKST() {
  try {
    return new Date().toLocaleString("ko-KR", { timeZone: "Asia/Seoul" });
  } catch {
    return new Date().toLocaleString();
  }
}

function setMeta(msg, tone = "") {
  const meta = $("meta");
  if (!meta) return;

  const toneClass = tone === "ok" ? "text-emerald-300" : tone === "err" ? "text-rose-300" : "text-slate-200";
  meta.innerHTML = `상태: <span class="${toneClass}">${msg}</span> <span class="text-xs text-slate-500">(${nowKST()})</span>`;
}

function lockUI(locked) {
  const ids = ["btnFetch", "btnReco", "market", "limit", "days", "asof"]; // optional inputs 포함
  for (const id of ids) {
    const el = $(id);
    if (!el) continue;
    el.disabled = !!locked;
    if (locked) el.classList.add("opacity-60", "pointer-events-none");
    else el.classList.remove("opacity-60", "pointer-events-none");
  }
}

function renderTable(items) {
  const wrap = $("tableWrap");
  if (!wrap) return;

  if (!items || items.length === 0) {
    wrap.innerHTML = `<div class="text-sm text-slate-400 p-3">추천 결과가 없습니다.</div>`;
    return;
  }

  const rows = items
    .map((it, idx) => {
      const rank = it.rank ?? (idx + 1);
      const ticker = it.ticker || it.symbol || it.code || "";
      const scoreVal = typeof it.score === "number" ? it.score : Number(it.score || 0);
      const score = Number.isFinite(scoreVal) ? scoreVal.toFixed(2) : "-";
      const note = it.note || it.reason || "";

      // 차트 API가 있으면 클릭으로 확인 가능
      const tickerCell = ticker
        ? `<a class="underline decoration-white/20 hover:decoration-white/60" href="/api/chart/${encodeURIComponent(ticker)}" target="_blank" rel="noreferrer">${ticker}</a>`
        : "";

      return `
        <tr class="border-b border-white/5 hover:bg-white/5">
          <td class="px-3 py-2 text-slate-300">${rank}</td>
          <td class="px-3 py-2 font-medium text-white">${tickerCell}</td>
          <td class="px-3 py-2 text-slate-300">${score}</td>
          <td class="px-3 py-2 text-xs text-slate-400">${note}</td>
        </tr>
      `;
    })
    .join("");

  wrap.innerHTML = `
    <table class="w-full text-sm">
      <thead class="sticky top-0 bg-slate-950/90 backdrop-blur border-b border-white/10">
        <tr class="text-left text-xs text-slate-400">
          <th class="px-3 py-2 w-14">#</th>
          <th class="px-3 py-2 w-28">Ticker</th>
          <th class="px-3 py-2 w-24">Score</th>
          <th class="px-3 py-2">Note</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

async function initHealth() {
  // /api/health
  if (elExists("health")) {
    try {
      const data = await api("/api/health");
      $("health").textContent = data.ok ? "OK" : "NG";
      $("health").className = data.ok ? "text-emerald-300" : "text-rose-300";
    } catch {
      $("health").textContent = "ERROR";
      $("health").className = "text-rose-300";
    }
  }

  // /api/health/db (있으면 표시)
  if (elExists("healthDb")) {
    try {
      const data = await api("/api/health/db");
      $("healthDb").textContent = data.ok ? "OK" : "NG";
      $("healthDb").className = data.ok ? "text-emerald-300" : "text-rose-300";
    } catch {
      $("healthDb").textContent = "ERROR";
      $("healthDb").className = "text-rose-300";
    }
  }
}

function getInputs() {
  const market = elExists("market") ? $("market").value : "ALL";
  const limit = elExists("limit") ? parseInt($("limit").value || "50", 10) : 50;

  // optional: days/asof inputs가 템플릿에 없으면 기본값 사용
  const days = elExists("days") ? parseInt($("days").value || "60", 10) : 60;
  const asofRaw = elExists("asof") ? ($("asof").value || "") : "";
  const asof = asofRaw.trim() ? asofRaw.trim() : null;

  return { market, limit, days, asof };
}

function bindUI() {
  const btnFetch = $("btnFetch");
  const btnReco = $("btnReco");

  if (btnFetch) {
    btnFetch.addEventListener("click", async () => {
      const { market, days, asof } = getInputs();

      lockUI(true);
      setMeta("DB 초기화/확인 중…");
      try {
        await api("/api/admin/init-db", { method: "POST" });

        // ingest: DB 기준으로 없는 데이터만 채우는 로직(서버)
        setMeta("데이터 가져오는 중… (없는 날짜만)");
        const qs = new URLSearchParams({ market, days: String(days) });
        if (asof) qs.set("asof", asof);

        const res = await api(`/api/admin/ingest?${qs.toString()}`, { method: "POST" });

        const td = res.trading_days_fetched ?? 0;
        const rows = res.rows_upserted_total ?? 0;
        setMeta(`수집 완료: 거래일 ${td}일 / rows ${rows}`, "ok");

        // 수집 후 DB health도 같이 갱신
        await initHealth();
      } catch (e) {
        setMeta(`에러: ${e.message}`, "err");
      } finally {
        lockUI(false);
      }
    });
  }

  if (btnReco) {
    btnReco.addEventListener("click", async () => {
      const { market, limit } = getInputs();

      lockUI(true);
      try {
        setMeta("추천 계산 중…");
        const qs = new URLSearchParams({ market, top_n: String(limit) });
        const screenRes = await api(`/api/admin/screen?${qs.toString()}`, { method: "POST" });

        setMeta("추천 불러오는 중…");
        const data = await api(`/api/reco/latest?market=${encodeURIComponent(market)}&limit=${limit}`);

        // 서버 응답 키가 환경마다 다를 수 있어 넉넉하게 대응
        const items = data.items ?? data.recos ?? data.results ?? data.top ?? [];
        renderTable(items);

        const run =
          data.run ??
          (data.run_id ? { id: data.run_id, asof_date: data.asof_date, market: data.market } : null) ??
          (screenRes?.run_id ? { id: screenRes.run_id, asof_date: screenRes.asof_date, market: screenRes.market } : null);

        const n = Array.isArray(items) ? items.length : 0;
        if (n > 0) {
          const rid = run?.id ?? "?";
          const asofTxt = run?.asof_date ? `, asof=${run.asof_date}` : "";
          setMeta(`추천 ${n}개 (run_id=${rid}${asofTxt})`, "ok");
        } else {
          setMeta("추천 결과 없음", "err");
        }
      } catch (e) {
        setMeta(`에러: ${e.message}`, "err");
      } finally {
        lockUI(false);
      }
    });
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  bindUI();
  await initHealth();
});