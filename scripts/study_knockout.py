#!/usr/bin/env python3
"""Wie stark bestraft die Knock-out-Schwelle einen kurzen Ausschlag?

Ein Turbo stirbt, sobald der Basiswert die Barriere BERUEHRT -- nicht erst,
wenn er darunter schliesst. Ein Modell, das nur Schlusskurse betrachtet,
unterschaetzt das Ausfallrisiko daher systematisch. Fuer einen driftfreien
Zufallspfad gilt nach dem Spiegelungsprinzip

    P(Barriere wird beruehrt) ~ 2 * P(Schlusskurs jenseits der Barriere)

Das Ausknockrisiko ist also rund doppelt so hoch wie eine reine
Schlusskursbetrachtung nahelegt. Dieses Skript misst den Effekt an echten
SG-Produkten und sucht den Punkt, an dem mehr Hebel sich nicht mehr lohnt.

    python3 scripts/study_knockout.py --asset Tesla --tagesvola 0.09
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy import rules                                              # noqa: E402
from strategy.sg_api import fx_to_eur, infer_spot, to_turbos, validate_pricing  # noqa: E402


def tagespfade(spot: float, tagesvola: float, n: int, schritte: int,
               df_t: float = 3.0, seed: int = 5) -> np.ndarray:
    """Intraday-Pfade eines Handelstags. schritte=78 entspricht 5-Minuten-Takt."""
    rng = np.random.default_rng(seed)
    z = rng.standard_t(df_t, size=(n, schritte)) / np.sqrt(df_t / (df_t - 2.0))
    return spot * np.cumprod(1.0 + tagesvola / np.sqrt(schritte) * z, axis=1)


def bewerte(t, pfade: np.ndarray, budget: float) -> dict:
    stueck = min(budget / t.ask, rules.MAX_LEVERAGE_UNITS)
    einsatz = stueck * t.ask
    S_end = pfade[:, -1]
    strike = t.strike if t.strike is not None else t.ko_barrier
    if t.direction > 0:
        beruehrt = pfade.min(axis=1) <= t.ko_barrier
        schluss_drunter = S_end <= t.ko_barrier
        wert = np.maximum(S_end - strike, 0.0)
        rest = max(t.ko_barrier - strike, 0.0)
    else:
        beruehrt = pfade.max(axis=1) >= t.ko_barrier
        schluss_drunter = S_end >= t.ko_barrier
        wert = np.maximum(strike - S_end, 0.0)
        rest = max(strike - t.ko_barrier, 0.0)
    sleeve = stueck * np.where(beruehrt, rest * t.ratio, wert * t.ratio)
    depot = (rules.START_CAPITAL - einsatz) + sleeve
    tag = depot / rules.START_CAPITAL - 1.0
    return {
        "wkn": t.wkn, "hebel": t.omega, "ko_abstand": t.distance_to_ko_pct,
        "spread": t.spread_pct, "einsatz": einsatz,
        "P_ko_intraday": float(beruehrt.mean()),
        "P_ko_nur_schluss": float(schluss_drunter.mean()),
        "median_tag": float(np.median(tag)),
        "p99_tag": float(np.percentile(tag, 99)),
        "P_tag_25": float((tag >= 0.25).mean()),
        "P_tag_50": float((tag >= 0.50).mean()),
        "P_tag_100": float((tag >= 1.00).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/sg/einzelaktien.csv")
    ap.add_argument("--asset", default="Tesla")
    ap.add_argument("--tagesvola", type=float, default=0.09,
                    help="Tagesbewegung des Basiswerts (0,09 = Quartalszahlen)")
    ap.add_argument("--pfade", type=int, default=60_000)
    ap.add_argument("--schritte", type=int, default=78, help="5-Minuten-Takt")
    args = ap.parse_args()

    recs = pd.read_csv(args.csv).query("_asset_name == @args.asset").to_dict("records")
    if not recs:
        print(f"Basiswert '{args.asset}' nicht in {args.csv}")
        return
    fx = fx_to_eur(recs)
    spot = infer_spot(recs, fx)
    ts = to_turbos(recs, verbose=False)
    ts, _ = validate_pricing(ts, spot, verbose=False)
    ts = [t for t in ts if t.product_type == "turbo" and t.spread_pct <= 0.05
          and t.direction > 0]

    pfade = tagespfade(spot, args.tagesvola, args.pfade, args.schritte)
    print(f"{args.asset} @ {spot:,.2f} EUR | Tagesvola {args.tagesvola*100:.1f} % | "
          f"{args.pfade:,} Pfade a {args.schritte} Intraday-Schritte\n")

    # je KO-Abstands-Band den Schein mit engstem Spread
    baender = [(0.005, 0.01), (0.01, 0.02), (0.02, 0.035), (0.035, 0.05),
               (0.05, 0.08), (0.08, 0.12), (0.12, 0.20), (0.20, 0.35)]
    zeilen = []
    for lo, hi in baender:
        kand = [t for t in ts if lo <= t.distance_to_ko_pct < hi]
        if not kand:
            continue
        zeilen.append(bewerte(min(kand, key=lambda t: t.spread_pct), pfade,
                              rules.leverage_budget(rules.START_CAPITAL)))

    print(f"{'KO-Abst':>8}{'Hebel':>7}{'Spread':>8}{'Einsatz':>9}"
          f"{'P(KO) intraday':>16}{'P(KO) nur Schluss':>19}{'Unter-':>8}")
    print(f"{'':>8}{'':>7}{'':>8}{'':>9}{'':>16}{'':>19}{'schaetzung':>11}")
    print("-" * 78)
    for z in zeilen:
        faktor = z["P_ko_intraday"] / max(z["P_ko_nur_schluss"], 1e-9)
        print(f"{z['ko_abstand']*100:>7.2f}%{z['hebel']:>7.1f}{z['spread']*100:>7.1f}%"
              f"{z['einsatz']:>9,.0f}{z['P_ko_intraday']:>15.1%}"
              f"{z['P_ko_nur_schluss']:>18.1%}{faktor:>10.2f}x")

    print(f"\n{'KO-Abst':>8}{'Hebel':>7}{'Median Tag':>12}{'p99 Tag':>10}"
          f"{'P>=+25%':>10}{'P>=+50%':>10}{'P>=+100%':>10}")
    print("-" * 67)
    for z in zeilen:
        print(f"{z['ko_abstand']*100:>7.2f}%{z['hebel']:>7.1f}{z['median_tag']*100:>11.1f}%"
              f"{z['p99_tag']*100:>9.1f}%{z['P_tag_25']:>10.2%}"
              f"{z['P_tag_50']:>10.2%}{z['P_tag_100']:>10.2%}")

    if zeilen:
        best = max(zeilen, key=lambda z: z["P_tag_50"])
        print(f"\nOptimum fuer P(Tag >= +50 %): KO-Abstand {best['ko_abstand']*100:.2f} %, "
              f"Hebel {best['hebel']:.1f} ({best['wkn']}) -> {best['P_tag_50']:.2%}")
        print("Enger ist nicht besser: unterhalb dieses Punktes frisst die")
        print("Barrierenberuehrung den Hebelvorteil vollstaendig auf.")


if __name__ == "__main__":
    main()
