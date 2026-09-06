"""Marktdatenzugriff ueber kostenlose Quellen.

yfinance faellt in dieser Umgebung aus (curl_cffi kommt nicht durch den Proxy),
deshalb ein schlanker eigener Client auf die oeffentliche Yahoo-Chart-API mit
`requests`. Liefert Historie, Intraday-Bars und die vordefinierten
Yahoo-Screener (day_gainers, most_actives, small_cap_gainers ...), die als
Momentum-Rohstoff dienen.

Hinweis zur Datenqualitaet: Diese Kurse sind gegenueber den EUWAX-Echtzeitkursen
des Spiels verzoegert. Sie taugen fuer Screening und Kalibrierung, nicht fuer
Sekunden-Timing beim Ordern.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")
_HOSTS = ["https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com"]

_session = requests.Session()
_session.headers.update({"User-Agent": _UA, "Accept": "application/json"})


def _get_json(path: str, params: dict | None = None, tries: int = 4) -> dict | None:
    last = None
    for attempt in range(tries):
        host = _HOSTS[attempt % len(_HOSTS)]
        try:
            r = _session.get(host + path, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
        except Exception as exc:                      # Netz/Proxy-Aussetzer
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(1.5 * (attempt + 1))
    print(f"  [data] {path} fehlgeschlagen: {last}")
    return None


# ------------------------------------------------------------------- Historie
def history(symbol: str, rng: str = "6mo", interval: str = "1d",
            cache_minutes: int = 30) -> pd.DataFrame:
    """OHLCV-Historie als DataFrame (leer, wenn der Abruf scheitert)."""
    cache = CACHE_DIR / f"{symbol.replace('^','_').replace('.','_')}_{rng}_{interval}.json"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < cache_minutes * 60:
        payload = json.loads(cache.read_text())
    else:
        payload = _get_json(f"/v8/finance/chart/{symbol}",
                            {"range": rng, "interval": interval,
                             "includePrePost": "false", "events": "div,split"})
        if payload is None:
            return pd.DataFrame()
        cache.write_text(json.dumps(payload))

    try:
        res = payload["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        df = pd.DataFrame({
            "open": q.get("open"), "high": q.get("high"), "low": q.get("low"),
            "close": q.get("close"), "volume": q.get("volume"),
        }, index=pd.to_datetime(res["timestamp"], unit="s", utc=True))
        df.attrs["currency"] = res["meta"].get("currency")
        df.attrs["exchange"] = res["meta"].get("fullExchangeName")
        df.attrs["last"] = res["meta"].get("regularMarketPrice")
        return df.dropna(subset=["close"])
    except (KeyError, IndexError, TypeError):
        return pd.DataFrame()


def last_price(symbol: str) -> float | None:
    payload = _get_json(f"/v8/finance/chart/{symbol}", {"range": "1d", "interval": "1d"})
    try:
        return float(payload["chart"]["result"][0]["meta"]["regularMarketPrice"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


# ------------------------------------------------------------------- Screener
SCREEN_IDS = ["day_gainers", "day_losers", "most_actives",
              "small_cap_gainers", "aggressive_small_caps", "undervalued_growth_stocks"]


@dataclass
class ScreenHit:
    symbol: str
    name: str
    price: float
    change_pct: float
    volume: float
    market_cap: float | None
    currency: str


def predefined_screen(scr_id: str = "day_gainers", count: int = 50) -> list[ScreenHit]:
    """Yahoos vordefinierte Screener -- unser taeglicher Momentum-Rohstoff."""
    payload = _get_json("/v1/finance/screener/predefined/saved",
                        {"scrIds": scr_id, "count": count, "start": 0})
    hits: list[ScreenHit] = []
    try:
        quotes = payload["finance"]["result"][0]["quotes"]
    except (KeyError, IndexError, TypeError):
        return hits
    for q in quotes:
        try:
            hits.append(ScreenHit(
                symbol=q["symbol"],
                name=q.get("shortName") or q.get("longName") or q["symbol"],
                price=float(q.get("regularMarketPrice") or 0.0),
                change_pct=float(q.get("regularMarketChangePercent") or 0.0),
                volume=float(q.get("regularMarketVolume") or 0.0),
                market_cap=(float(q["marketCap"]) if q.get("marketCap") else None),
                currency=q.get("currency", "USD"),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return hits


# --------------------------------------------------------------- Marktkontext
BENCHMARKS = {
    "DAX": "^GDAXI", "S&P 500": "^GSPC", "Nasdaq 100": "^NDX",
    "VIX": "^VIX", "VDAX-NEW": "^VDAX", "EUR/USD": "EURUSD=X",
    "Gold": "GC=F", "Brent": "BZ=F", "US 10Y": "^TNX",
}


def market_snapshot() -> pd.DataFrame:
    """Aktueller Stand der Leitmaerkte inkl. realisierter 20-Tage-Vola."""
    rows = []
    for label, sym in BENCHMARKS.items():
        df = history(sym, rng="3mo", interval="1d")
        if df.empty or len(df) < 21:
            rows.append({"Markt": label, "Symbol": sym, "Kurs": None,
                         "1T %": None, "20T %": None, "Vola 20T ann.": None})
            continue
        c = df["close"]
        rows.append({
            "Markt": label, "Symbol": sym, "Kurs": round(float(c.iloc[-1]), 2),
            "1T %": round(float(c.pct_change().iloc[-1] * 100), 2),
            "20T %": round(float((c.iloc[-1] / c.iloc[-21] - 1) * 100), 2),
            "Vola 20T ann.": round(float(c.pct_change().tail(20).std() * (252 ** 0.5) * 100), 1),
        })
    return pd.DataFrame(rows)
