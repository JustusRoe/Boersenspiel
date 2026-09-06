#!/usr/bin/env python3
"""Misst empirisch, wie stark sich Indizes an FOMC-Entscheidungstagen bewegen.

Statt eine Ereignis-Volatilitaet zu raten, wird sie aus der Historie
geschaetzt. Wichtige Feinheit fuer das Boersenspiel: Der FOMC-Entscheid faellt
um 20:00 MEZ. Die US-Indizes schliessen um 22:00 MEZ und bilden die Reaktion
daher am selben Tag ab. Der DAX-Kassaindex schliesst dagegen um 17:30 -- seine
offizielle Schlussnotierung enthaelt die Reaktion NICHT, sie erscheint erst am
Folgetag. SG bepreist DAX-Turbos nach Xetra-Schluss ueber den Future, das
Produkt bewegt sich also sehr wohl; nur der Index selbst hinkt nach.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy import data  # noqa: E402

# Zweiter Sitzungstag = Tag der Entscheidung
FOMC_DECISION_DAYS = [
    date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7), date(2025, 6, 18),
    date(2025, 7, 30), date(2025, 9, 17), date(2025, 10, 29), date(2025, 12, 10),
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29), date(2026, 6, 17),
    date(2026, 7, 29),
]

MAERKTE = {"S&P 500": "^GSPC", "Nasdaq 100": "^NDX", "DAX": "^GDAXI"}


def main() -> None:
    print("Bewegung an FOMC-Entscheidungstagen (Schluss zu Schluss)")
    print("=" * 78)
    ergebnis = {}
    for label, sym in MAERKTE.items():
        df = data.history(sym, rng="2y", interval="1d")
        if df.empty:
            print(f"{label}: keine Daten")
            continue
        r = df["close"].pct_change().dropna()
        r.index = [d.date() for d in r.index]
        s = pd.Series(r.values, index=r.index)

        treffer, folgetag = [], []
        alle_tage = list(s.index)
        for d in FOMC_DECISION_DAYS:
            if d in s.index:
                treffer.append(float(s.loc[d]))
                i = alle_tage.index(d)
                if i + 1 < len(alle_tage):
                    folgetag.append(float(s.iloc[i + 1]))
        if not treffer:
            print(f"{label}: keine FOMC-Tage in der Historie")
            continue

        andere = s[~s.index.isin(FOMC_DECISION_DAYS)]
        a = np.array(treffer)
        ergebnis[label] = {"fomc": a, "folgetag": np.array(folgetag),
                           "normal": andere.values}

        print(f"\n{label}  ({len(a)} FOMC-Tage in 2 Jahren)")
        print(f"  FOMC-Tag      |Bewegung| Median {np.median(np.abs(a))*100:5.2f} %   "
              f"Mittel {np.mean(np.abs(a))*100:5.2f} %   Max {np.max(np.abs(a))*100:5.2f} %")
        print(f"  normaler Tag  |Bewegung| Median {np.median(np.abs(andere))*100:5.2f} %   "
              f"Mittel {np.mean(np.abs(andere))*100:5.2f} %")
        faktor = np.mean(np.abs(a)) / np.mean(np.abs(andere))
        print(f"  -> Ereignis-Aufschlag: Faktor {faktor:.2f}")
        vol_fomc = float(np.std(a) * np.sqrt(252))
        print(f"  -> implizierte Tagesvola (annualisiert): {vol_fomc*100:.1f} %")
        if len(folgetag):
            f = np.array(folgetag)
            print(f"  Folgetag      |Bewegung| Median {np.median(np.abs(f))*100:5.2f} %")

    if "DAX" in ergebnis:
        print("\n" + "=" * 78)
        print("Wichtig fuer die Instrumentenwahl:")
        d = ergebnis["DAX"]
        print(f"  DAX am FOMC-Tag selbst: {np.mean(np.abs(d['fomc']))*100:.2f} % "
              f"| am Folgetag: {np.mean(np.abs(d['folgetag']))*100:.2f} %")
        print("  Der Kassaindex schliesst um 17:30, also vor dem Entscheid um 20:00.")
        print("  Fuer eine Tageswette auf den Entscheid sind US-Index-Produkte")
        print("  sauberer -- dort faellt der Bewertungsschluss mit dem Ereignis")
        print("  zusammen. Bei DAX-Turbos haengt alles daran, ob SG nach 17:30")
        print("  ueber den Future weiterstellt (Standardpraxis, am 10.09. zur")
        print("  EZB pruefbar).")


if __name__ == "__main__":
    main()
