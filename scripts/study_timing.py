#!/usr/bin/env python3
"""Wann im Spielverlauf soll das Sniper-Depot die konvexe Wette eingehen?

Die Vermutung, frueh sei besser, weil noch mehr Ruecksetzer folgen, laesst
sich pruefen. Dagegen steht ein Effekt, der leicht uebersehen wird: Nach
einem Treffer ist das Hebelbudget bei absolut 20.000 EUR gedeckelt und faellt
als Anteil am Depot steil ab. Ein Depot bei 400.000 EUR kann nur noch 5 %
gehebelt anlegen. Weiter waechst dann nur der Aktienteil.

Simuliert wird woechentlich: In einer aggressiven Woche liegen 20.000 EUR in
einem Turbo mit enger Barriere, der Rest in beweglichen Aktien. In einer
ruhigen Woche liegt alles in Aktien. Montags wird zurueckgesetzt, wenn der
Wert unter 100.000 EUR liegt.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy import rules  # noqa: E402

N = 60_000
WOCHEN = 8
TAGE_JE_WOCHE = 5
SCHRITTE = 26                      # Intraday-Schritte, Barriere zaehlt beruehrend

SIGMA_BASIS = 0.0455               # AMD, typische Tagesbewegung
KO_ABSTAND = 0.0391                # FG619W
HEBEL = 22.5
SIGMA_AKTIE = 0.045                # Aktien-Sleeve, je Position
N_AKTIEN = 4


def spiele(aggressive_wochen: set[int], seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    wert = np.full(N, rules.START_CAPITAL)
    for w in range(1, WOCHEN + 1):
        # Montag: Ruecksetzer, wenn unter Startwert
        wert = np.where(wert < rules.START_CAPITAL, rules.START_CAPITAL, wert)

        budget = np.minimum(rules.MAX_LEVERAGE_WEIGHT * wert, rules.MAX_LEVERAGE_EUR)
        if w not in aggressive_wochen:
            budget = np.zeros(N)
        aktien = wert - budget

        # Hebel-Sleeve ueber die Woche, Barriere intraday geprueft
        z = rng.standard_t(3.0, size=(N, TAGE_JE_WOCHE * SCHRITTE)) / math.sqrt(3.0)
        schritt = 1 + SIGMA_BASIS / math.sqrt(SCHRITTE) * z
        pfad = np.cumprod(schritt, axis=1)
        ko = pfad.min(axis=1) <= 1 - KO_ABSTAND
        endwert = np.maximum(pfad[:, -1] - (1 - KO_ABSTAND), 0.0) / KO_ABSTAND
        sleeve = np.where(ko, 0.0, budget * endwert)

        # Aktienteil: vier Positionen, je eigene Bewegung
        r = np.zeros(N)
        for _ in range(N_AKTIEN):
            zz = rng.standard_t(3.0, size=(N, TAGE_JE_WOCHE)) / math.sqrt(3.0)
            r += (1.0 / N_AKTIEN) * (np.prod(1 + SIGMA_AKTIE * zz, axis=1) - 1)
        wert = sleeve + aktien * (1 + r)
    return wert


def zeige(label: str, v: np.ndarray) -> None:
    print(f"{label:<34}{np.median(v):>10,.0f}{np.percentile(v, 99):>11,.0f}"
          f"{np.percentile(v, 99.9):>12,.0f}{(v >= 250_000).mean():>10.2%}"
          f"{(v >= 500_000).mean():>10.2%}{(v < 100_000).mean():>9.1%}")


def main() -> None:
    print(f"Endwert des Depots nach {WOCHEN} Wochen, {N:,} Pfade")
    print(f"Hebelwette: AMD-Turbo Hebel {HEBEL:.0f}, Barriere {KO_ABSTAND*100:.2f} % entfernt\n")
    print(f"{'Aggressive Wochen':<34}{'Median':>10}{'p99':>11}{'p99.9':>12}"
          f"{'P>=250k':>10}{'P>=500k':>10}{'P<100k':>9}")
    print("-" * 96)
    zeige("alle 8 Wochen", spiele(set(range(1, 9))))
    zeige("nur Wochen 1-4 (frueh)", spiele({1, 2, 3, 4}))
    zeige("nur Wochen 5-8 (spaet)", spiele({5, 6, 7, 8}))
    zeige("nur Woche 1", spiele({1}))
    zeige("nur Woche 8", spiele({8}))
    zeige("keine (nur Aktien)", spiele(set()))
    print()
    print("Der Vergleich frueh gegen spaet isoliert den Timing-Effekt: gleiche")
    print("Anzahl Versuche, nur zu anderen Zeitpunkten.")


if __name__ == "__main__":
    main()
