"""Screener fuer den Aktien-Sleeve (die 80 % ausserhalb der Hebel-Kappe).

Gesucht werden nicht 'gute Unternehmen', sondern Titel mit maximaler
rechtsseitiger Beweglichkeit: hohe realisierte Vola, Gap-Neigung, enger Float,
frischer Momentumausbruch. Jede Position darf 20 % des Depots ausmachen -- ein
Titel mit +100 % an einem Tag bringt damit +20 % Depotperformance.

Handelbarkeitsfilter: Boerse Stuttgart, Kurs >= 1,00 EUR zum Kaufzeitpunkt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import data

# Kern-Universum: an der Boerse Stuttgart breit handelbare Titel mit
# ueberdurchschnittlicher Beweglichkeit. Wird zur Laufzeit um die
# Yahoo-Tagesgewinner ergaenzt.
SEED_UNIVERSE = [
    # deutsche/europaeische Nebenwerte mit hoher Vola
    "RHM.DE", "NDA.DE", "EVT.DE", "SDF.DE", "TMV.DE", "AIXA.DE", "SHL.DE",
    "PNE3.DE", "VBK.DE", "NA9.DE", "ECV.DE", "B4B.DE", "COK.DE", "WAF.DE",
    # US-Hochvola-Namen (in Stuttgart handelbar)
    "TSLA", "PLTR", "COIN", "MSTR", "SMCI", "AFRM", "RIVN", "SOFI",
    "MARA", "RIOT", "CLSK", "IONQ", "RGTI", "LUNR", "ASTS", "OKLO",
    "NVDA", "AMD", "AVGO", "MU", "ARM", "CRWV",
]


@dataclass
class Candidate:
    symbol: str
    price: float
    currency: str
    ret_1d: float
    ret_5d: float
    ret_20d: float
    vol_20d_ann: float
    atr_pct: float
    rel_volume: float
    dist_52w_high: float
    upside_tail: float          # 95. Perzentil der Tagesrenditen (1 Jahr)
    convexity_score: float

    def as_row(self) -> dict:
        return {
            "Symbol": self.symbol, "Kurs": round(self.price, 2), "Waehr": self.currency,
            "1T %": round(self.ret_1d * 100, 1), "5T %": round(self.ret_5d * 100, 1),
            "20T %": round(self.ret_20d * 100, 1),
            "Vola ann %": round(self.vol_20d_ann * 100, 0),
            "ATR %": round(self.atr_pct * 100, 1),
            "RelVol": round(self.rel_volume, 2),
            "z. 52W-Hoch %": round(self.dist_52w_high * 100, 1),
            "Tail p95 %": round(self.upside_tail * 100, 1),
            "Score": round(self.convexity_score, 2),
        }


def _metrics(symbol: str) -> Candidate | None:
    df = data.history(symbol, rng="1y", interval="1d")
    if df.empty or len(df) < 60:
        return None
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    r = c.pct_change().dropna()
    if r.empty:
        return None

    price = float(c.iloc[-1])
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_pct = float((tr.tail(14).mean()) / price) if price else 0.0
    rel_vol = float(v.tail(5).mean() / v.tail(60).mean()) if v.tail(60).mean() else 1.0
    vol20 = float(r.tail(20).std() * math.sqrt(252))
    tail = float(np.percentile(r.tail(252), 95))
    dist_high = float(price / c.tail(252).max() - 1.0)

    # Konvexitaets-Score: belohnt Beweglichkeit und rechte Verteilungsschulter,
    # nicht Qualitaet oder Bewertung. Naehe zum 52-Wochen-Hoch als Momentumfilter.
    score = (2.5 * tail * 100
             + 1.0 * vol20 * 100 / 10
             + 1.5 * max(rel_vol - 1.0, 0) * 10
             + 1.0 * max(0.0, 1.0 + dist_high * 5) * 5
             + 0.8 * atr_pct * 100)

    return Candidate(symbol, price, df.attrs.get("currency", "?"),
                     float(r.iloc[-1]),
                     float(c.iloc[-1] / c.iloc[-6] - 1) if len(c) > 6 else 0.0,
                     float(c.iloc[-1] / c.iloc[-21] - 1) if len(c) > 21 else 0.0,
                     vol20, atr_pct, rel_vol, dist_high, tail, score)


def scan(extra_symbols: list[str] | None = None, include_yahoo_screens: bool = True,
         max_symbols: int = 70, min_price_eur: float = 1.0,
         eurusd: float | None = None) -> pd.DataFrame:
    """Baut das Kandidatenuniversum und bewertet es."""
    symbols = list(dict.fromkeys(SEED_UNIVERSE + (extra_symbols or [])))

    if include_yahoo_screens:
        for scr in ("day_gainers", "small_cap_gainers", "most_actives"):
            for hit in data.predefined_screen(scr, count=25):
                if hit.symbol not in symbols and hit.price >= 1.0:
                    symbols.append(hit.symbol)

    if eurusd is None:
        eurusd = data.last_price("EURUSD=X") or 1.16

    rows, used = [], 0
    for sym in symbols:
        if used >= max_symbols:
            break
        cand = _metrics(sym)
        if cand is None:
            continue
        px_eur = cand.price / eurusd if cand.currency == "USD" else cand.price
        if px_eur < min_price_eur:          # Regel: Mindestkurs 1,00 EUR
            continue
        row = cand.as_row()
        row["Kurs EUR"] = round(px_eur, 2)
        rows.append(row)
        used += 1

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    return df
