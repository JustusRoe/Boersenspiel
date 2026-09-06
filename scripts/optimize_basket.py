#!/usr/bin/env python3
"""Optimiert den Hebelkorb auf echten SG-Produktdaten.

Voraussetzung: scripts/sg_scrape.py wurde ausgefuehrt.

    python3 scripts/sg_scrape.py --assets "DAX 40" --typen turbo
    python3 scripts/optimize_basket.py --depot 100000 --horizont 5

Der Optimizer maximiert P(Depot >= Ziel), nicht den Erwartungswert -- im
Boersenspiel gibt es keinen Preis fuer einen guten Median.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy import data, rules                              # noqa: E402
from strategy.optimizer import optimise_basket                # noqa: E402
from strategy.sg_api import to_turbos, validate_pricing       # noqa: E402

VOLA_PROXY = {"DAX 40": "^GDAXI", "Nasdaq-100": "^NDX", "S&P 500": "^GSPC"}


def realisierte_vola(symbol: str, tage: int = 20) -> float | None:
    df = data.history(symbol, rng="6mo", interval="1d")
    if df.empty or len(df) < tage + 1:
        return None
    return float(df["close"].pct_change().tail(tage).std() * (252 ** 0.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/sg/sg_products.csv")
    ap.add_argument("--depot", type=float, default=100_000.0)
    ap.add_argument("--horizont", type=int, default=5, help="Handelstage")
    ap.add_argument("--ziel", type=float, default=1.5, help="Zielvielfaches des Depots")
    ap.add_argument("--asset", help="auf einen Basiswert einschraenken")
    ap.add_argument("--max-spread", type=float, default=0.06,
                    help="Produkte mit groesserem Geld-Brief-Spread verwerfen")
    ap.add_argument("--vola-aufschlag", type=float, default=1.25,
                    help="Faktor auf die realisierte Vola (implizite liegt darueber)")
    ap.add_argument("--pfade", type=int, default=30_000)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.asset:
        df = df[df["_asset_name"].str.contains(args.asset, case=False, na=False)]
    if df.empty:
        print("Keine Produkte nach Filterung.")
        return

    vol_by_underlying: dict[str, float] = {}
    produkte = []
    for asset_name, grp in df.groupby("_asset_name"):
        spot = float(grp["_spot"].iloc[0])
        ts = to_turbos(grp.to_dict("records"), spot=spot, verbose=False)
        vor = len(ts)
        # Kreuzprobe gegen die Theorie, bevor irgendetwas optimiert wird
        ts, _ = validate_pricing(ts, spot, verbose=False)
        nach_probe = len(ts)
        ts = [t for t in ts if t.spread_pct <= args.max_spread]
        produkte.extend(ts)

        sym = VOLA_PROXY.get(asset_name)
        rv = realisierte_vola(sym) if sym else None
        vol = (rv or 0.20) * args.vola_aufschlag
        vol_by_underlying[asset_name] = vol
        print(f"  {asset_name:<12} {len(grp):>5,} Zeilen -> {vor:>5,} bewertbar -> "
              f"{nach_probe:>5,} nach Preisprobe -> {len(ts):>5,} nach Spread | "
              f"Kurs {spot:,.2f} | Vola {(rv or 0)*100:4.1f}% -> {vol*100:4.1f}%")

    if not produkte:
        print("Keine bewertbaren Produkte uebrig.")
        return

    print(f"\nOptimiere {len(produkte):,} Produkte | Depot {args.depot:,.0f} EUR | "
          f"Hebelbudget {rules.leverage_budget(args.depot):,.0f} EUR | "
          f"Horizont {args.horizont} Handelstage")

    korb = optimise_basket(produkte, args.depot, args.horizont, vol_by_underlying,
                           objective="p_target", target_multiple=args.ziel,
                           n_paths=args.pfade, n_restarts=300)
    print()
    print(korb.to_table())
    print()
    for k, v in korb.stats.items():
        print(f"   {k:32s} {v:,.4f}" if isinstance(v, float) else f"   {k:32s} {v}")

    gewaehlt = [p for p in korb.products if korb.units.get(p.wkn, 0) > 0]
    if gewaehlt:
        print("\n   Details der gewaehlten Scheine:")
        for p in gewaehlt:
            print(f"     {p.wkn}  {p.underlying:<12} {'Long' if p.direction > 0 else 'Short':<5} "
                  f"Geld {p.bid:>8.3f} Brief {p.ask:>8.3f} "
                  f"Spread {p.spread_pct*100:>4.1f}% Hebel {p.omega:>5.1f} "
                  f"KO-Abstand {p.distance_to_ko_pct*100:>5.2f}%")


if __name__ == "__main__":
    main()
