#!/usr/bin/env python3
"""Sucht das Derivat mit der groessten moeglichen Tagesbewegung der 20.000 EUR.

Zwei Grenzen wirken gleichzeitig: 20.000 EUR und 20.000 Stueck. Damit die
EUR-Grenze bindet und nicht die Stueck-Grenze, muss der Schein mindestens
1,00 EUR kosten -- darunter laesst sich das Budget mit einem einzigen Produkt
gar nicht ausschoepfen.

Innerhalb dieser Bedingung zaehlt nicht der Hebel allein, sondern
Hebel x Tagesvolatilitaet des Basiswerts: So weit bewegen sich die 20.000 EUR
an einem Tag. Zusaetzlich ausgewiesen wird das Sprungpotenzial, denn fuer die
beste Tagesperformance braucht es nicht den gleichmaessigen Tag, sondern den
einen aussergewoehnlichen.

    python3 scripts/finde_tagesschein.py --min-preis 1.00
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy import data, rules                                    # noqa: E402
from strategy.sg_api import (CLS_BEST_TURBO_OPEN_END,               # noqa: E402
                             CLS_UNLIMITED_TURBO_MINI, SGClient,
                             fx_to_eur, infer_spot, to_turbos, validate_pricing)

# Basiswerte mit hoher Bewegung aus den bisherigen Scans, plus Yahoo-Symbol
KANDIDATEN = {
    "NVIDIA": (320, "NVDA"), "Advanced Micro Devices (AMD)": (214, "AMD"),
    "Micron Technology": (171, "MU"), "Tesla": (72, "TSLA"),
    "Palantir Technologies": (1673, "PLTR"), "Coinbase Global": (3999, "COIN"),
    "Strategy": (4530, "MSTR"), "CrowdStrike Holdings": (107, "CRWD"),
    "Broadcom": (641, "AVGO"), "Microsoft": (331, "MSFT"),
    "Intel": (250, "INTC"), "Robinhood Markets": (4919, "HOOD"),
    "DAX 40": (389, "^GDAXI"),
}


def vola(sym: str) -> dict | None:
    h = data.history(sym, rng="1y", interval="1d", cache_minutes=600)
    if h.empty or len(h) < 120:
        return None
    r = h["close"].pct_change().dropna()
    r60 = r.tail(60)
    typisch = float(np.median(np.abs(r60)) * 1.4826)
    return {"typisch": typisch, "std": float(r60.std()),
            "sprungtage": float((r.tail(252).abs() > 0.08).mean() * 252),
            "max_tag": float(r.tail(252).abs().max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-preis", type=float, default=1.00,
                    help="Mindest-Briefkurs, damit die EUR-Grenze bindet")
    ap.add_argument("--max-spread", type=float, default=0.025)
    ap.add_argument("--richtung", choices=["long", "short", "beide"], default="beide")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    budget = rules.leverage_budget(rules.START_CAPITAL)
    clients = [SGClient(pause=0.4 * args.workers) for _ in range(args.workers)]

    def hole(i_item):
        i, (name, (aid, sym)) = i_item
        cl = clients[i % args.workers]
        recs = []
        for cls in (CLS_BEST_TURBO_OPEN_END, CLS_UNLIMITED_TURBO_MINI):
            try:
                recs += cl.search(cls, aid, page_size=500, page_number=1).get("Products", [])
            except Exception:
                continue
        if not recs:
            return None
        v = vola(sym)
        if not v:
            return None
        fx = fx_to_eur(recs)
        spot = infer_spot(recs, fx)
        ts = to_turbos(recs, spot=spot, verbose=False)
        ts, _ = validate_pricing(ts, spot, verbose=False)
        treffer = []
        for t in ts:
            if t.product_type != "turbo" or t.ask < args.min_preis:
                continue
            if t.spread_pct > args.max_spread or t.spread_pct < 0:
                continue          # negative Spreads sind Quote-Artefakte
            if args.richtung == "long" and t.direction < 0:
                continue
            if args.richtung == "short" and t.direction > 0:
                continue
            stueck = min(int(budget // t.ask), rules.MAX_LEVERAGE_UNITS)
            einsatz = stueck * t.ask
            if einsatz < 0.97 * budget:      # Budget nicht ausschoepfbar
                continue
            treffer.append({
                "Basiswert": name, "WKN": t.wkn,
                "Richtung": "Long" if t.direction > 0 else "Short",
                "Hebel": t.omega, "Brief": t.ask, "Stueck": stueck,
                "Einsatz": einsatz, "Spread %": t.spread_pct * 100,
                "KO-Abstand %": t.distance_to_ko_pct * 100,
                "KO in Sigma": t.distance_to_ko_pct / v["typisch"],
                "Vola typisch %": v["typisch"] * 100,
                "Sprungtage/Jahr": v["sprungtage"], "groesster Tag %": v["max_tag"] * 100,
                "Sleeve-Tagesvola %": t.omega * v["typisch"] * 100})
        if not treffer:
            return None
        return max(treffer, key=lambda x: x["Sleeve-Tagesvola %"])

    zeilen = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for res in pool.map(hole, enumerate(KANDIDATEN.items())):
            if res:
                zeilen.append(res)

    if not zeilen:
        print("Nichts gefunden.")
        return
    df = pd.DataFrame(zeilen).sort_values("Sleeve-Tagesvola %", ascending=False)
    pd.set_option("display.width", 220)
    print(f"Bester Schein je Basiswert (Brief >= {args.min_preis:.2f} EUR, "
          f"Budget {budget:,.0f} EUR ausschoepfbar)\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    b = df.iloc[0]
    print(f"\nSpitzenreiter: {b['Basiswert']} {b['WKN']} {b['Richtung']}, Hebel {b['Hebel']:.0f}")
    print(f"  {b['Stueck']:,.0f} Stueck a {b['Brief']:.3f} = {b['Einsatz']:,.0f} EUR")
    print(f"  Bewegt sich am typischen Tag um {b['Sleeve-Tagesvola %']:.0f} %, also "
          f"{budget * b['Sleeve-Tagesvola %'] / 100:,.0f} EUR")
    print(f"  Barriere {b['KO-Abstand %']:.2f} % entfernt = {b['KO in Sigma']:.1f} "
          f"typische Tagesbewegungen")
    df.to_csv("data/tagesschein_kandidaten.csv", index=False)


if __name__ == "__main__":
    main()
