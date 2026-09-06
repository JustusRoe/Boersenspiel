#!/usr/bin/env python3
"""Lohnt es sich, an einem Ereignistag ein Depot Long und eines Short zu stellen?

Der Preis fuer die beste Tagesperformance wird als MAXIMUM ueber alle Depots
und alle Spieltage vergeben, nicht als Summe. Das aendert die Rechnung
grundlegend: Man braucht nicht ein Depot, das im Mittel gut laeuft, sondern
mindestens eines, das an einem einzigen Tag explodiert.

Zwei Eigenheiten der Spielregeln greifen dabei ineinander:

  1. Die 20.000-EUR-Hebelkappe gilt JE DEPOT. Zwei Depots bedeuten also
     40.000 EUR Hebeleinsatz, nicht 2 x 10.000 -- man teilt kein Budget.
  2. Der woechentliche Ruecksetzer erstattet das Verlierer-Depot. Wer mit
     100.000 startet, 20.000 in Turbos verliert und bei 80.000 landet, setzt
     am Montag auf 100.000 zurueck. Die Fehlwette kostet dann effektiv nichts.

Aufruf:
    python3 scripts/study_event_day.py --csv data/sg/dax.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy import rules                                   # noqa: E402
from strategy.optimizer import payoff_per_unit                # noqa: E402
from strategy.sg_api import to_turbos, validate_pricing       # noqa: E402


def tagesbewegungen(n: int, tagesvola: float, df_t: float = 3.0,
                    seed: int = 5) -> np.ndarray:
    """Tagesrenditen des Basiswerts mit fetten Raendern, driftfrei."""
    rng = np.random.default_rng(seed)
    z = rng.standard_t(df_t, size=n) / np.sqrt(df_t / (df_t - 2.0))
    return tagesvola * z


def sleeve_wert(t, spot: float, moves: np.ndarray, einsatz: float) -> np.ndarray:
    """Wert des Hebel-Sleeves am Tagesende, inkl. Knock-out und Stueck-Kappe."""
    stueck = min(einsatz / t.ask, rules.MAX_LEVERAGE_UNITS)
    S_end = spot * (1.0 + moves)
    # Intraday-Extrem grob mitnehmen: der Tagesweg geht ueber den Schluss hinaus
    S_min = np.minimum(S_end, spot * (1.0 + moves - np.abs(moves) * 0.6))
    S_max = np.maximum(S_end, spot * (1.0 + moves + np.abs(moves) * 0.6))
    return stueck * payoff_per_unit(t, S_end, S_min, S_max)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/sg/dax.csv")
    ap.add_argument("--tagesvola", type=float, default=0.0054,
                    help="Tagesbewegung des Basiswerts (Standardabw.), "
                         "empirisch fuer FOMC-Tage ~0,54 %% beim S&P")
    ap.add_argument("--aktien-vola", type=float, default=0.030,
                    help="Tagesvola je Aktienposition im 80-%%-Sleeve")
    ap.add_argument("--pfade", type=int, default=200_000)
    ap.add_argument("--ziel-hebel", type=float, default=80.0)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    asset = df["_asset_name"].iloc[0]
    spot = float(df["_spot"].iloc[0])
    ts = to_turbos(df.to_dict("records"), spot=spot, verbose=False)
    ts, _ = validate_pricing(ts, spot, verbose=False)
    ts = [t for t in ts if t.product_type == "turbo" and t.spread_pct <= 0.05]

    def bester(direction: int):
        """Schein mit dem Hebel am Zielwert, bei engstem Spread."""
        k = [t for t in ts if t.direction == direction]
        if not k:
            return None
        return min(k, key=lambda t: (abs(t.omega - args.ziel_hebel), t.spread_pct))

    lg, sh = bester(+1), bester(-1)
    if not lg or not sh:
        print("Kein passendes Long/Short-Paar gefunden.")
        return

    print(f"Basiswert {asset} @ {spot:,.2f} | Tagesvola {args.tagesvola*100:.2f} % | "
          f"{args.pfade:,} Pfade")
    for lab, t in (("Long ", lg), ("Short", sh)):
        print(f"  {lab}: {t.wkn} Hebel {t.omega:5.1f} KO-Abstand "
              f"{t.distance_to_ko_pct*100:5.2f} % Spread {t.spread_pct*100:4.1f} % "
              f"Brief {t.ask:.3f}")

    moves = tagesbewegungen(args.pfade, args.tagesvola, seed=5)
    rng = np.random.default_rng(9)
    budget = rules.leverage_budget(rules.START_CAPITAL)
    aktien = rules.START_CAPITAL - budget

    def depot_tagesrendite(t, aktien_seed):
        sleeve = sleeve_wert(t, spot, moves, budget)
        # 4 Aktien a 20 %, marktkorreliert plus Eigenbewegung
        r = np.zeros(args.pfade)
        g = np.random.default_rng(aktien_seed)
        for _ in range(4):
            eig = args.aktien_vola * g.standard_t(3.0, args.pfade) / np.sqrt(3.0)
            r += 0.25 * (1.1 * moves + eig)
        wert = sleeve + aktien * (1.0 + r)
        return wert / rules.START_CAPITAL - 1.0

    a_long = depot_tagesrendite(lg, 101)
    b_long = depot_tagesrendite(lg, 202)
    b_short = depot_tagesrendite(sh, 202)

    varianten = {
        "beide Long":        np.maximum(a_long, b_long),
        "Long + Short":      np.maximum(a_long, b_short),
        "nur ein Depot Long": a_long,
    }

    print(f"\n{'Aufstellung':<22}{'Median':>9}{'p90':>9}{'p99':>9}{'p99.9':>10}"
          f"{'P>=+25%':>10}{'P>=+50%':>10}{'P>=+100%':>10}")
    print("-" * 89)
    for lab, v in varianten.items():
        print(f"{lab:<22}{np.median(v)*100:>8.1f}%{np.percentile(v,90)*100:>8.1f}%"
              f"{np.percentile(v,99)*100:>8.1f}%{np.percentile(v,99.9)*100:>9.1f}%"
              f"{np.mean(v>=0.25):>10.2%}{np.mean(v>=0.50):>10.2%}"
              f"{np.mean(v>=1.00):>10.2%}")

    v1, v2 = varianten["beide Long"], varianten["Long + Short"]
    print(f"\nP(>=+25 %): Long+Short {np.mean(v2>=0.25):.2%} gegen "
          f"beide Long {np.mean(v1>=0.25):.2%} "
          f"-> Faktor {np.mean(v2>=0.25)/max(np.mean(v1>=0.25),1e-9):.2f}")

    # Was kostet die Fehlwette wirklich?
    verlierer = np.minimum(a_long, b_short)
    print(f"\nKosten der Fehlwette: Verlierer-Depot im Median bei "
          f"{(1+np.median(verlierer))*rules.START_CAPITAL:,.0f} EUR")
    print(f"  Ruecksetzer am Montag stellt 100.000 EUR her -> effektive Kosten "
          f"{max(0.0, -np.median(verlierer))*rules.START_CAPITAL:,.0f} EUR entfallen,")
    print("  bezahlt wird stattdessen mit einem der beiden Wochen-Ruecksetzer.")


if __name__ == "__main__":
    main()
