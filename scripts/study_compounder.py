#!/usr/bin/env python3
"""Welches Hebelprodukt traegt das Compounder-Depot ueber acht Wochen?

Ein Faktor-Zertifikat setzt den Hebel taeglich zurueck. Sein Logarithmus
entwickelt sich mit

    log-Rendite ~ L*mu*T - (L*sigma)^2 * T / 2

Der erste Term waechst linear im Hebel, der zweite quadratisch: Ab einem
gewissen Punkt frisst der Volatilitaetsdrag mehr, als der Hebel bringt. Fuer
ein Zielvielfaches M ueber T Tage folgt daraus ein Optimum in geschlossener
Form. Mit x = L*sigma*sqrt(T) ist

    P(Endwert >= M) = P(z >= log(M)/x + x/2)

und die Schwelle wird minimal bei x = sqrt(2*log(M)), also

    L* = sqrt(2*log(M)) / (sigma_taeglich * sqrt(T))

Das ist der praktisch wichtigste Punkt der ganzen Produktwahl: Mehr Hebel ist
bei Faktor-Zertifikaten NICHT besser -- anders als beim Knock-out-Turbo, wo
der Verlust bei null endet und die Ruecksetzer-Option ihn erstattet.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy import rules  # noqa: E402

T = 40           # Handelstage des Spiels
N = 200_000
EINSATZ = rules.leverage_budget(rules.START_CAPITAL)

# Tagesvola aus dem Universumsscan (20 Handelstage realisiert)
BASISWERTE = {
    "DAX 40": 0.0056, "NVIDIA": 0.0287, "Palantir": 0.0340,
    "Tesla": 0.0320, "CrowdStrike": 0.0607, "Microsoft": 0.0147,
}
# Bei SG tatsaechlich angebotene Faktor-Hebel
VERFUEGBAR = {"DAX 40": 15, "NVIDIA": 15, "Palantir": 14,
              "Tesla": 15, "CrowdStrike": 10, "Microsoft": 15}


def optimaler_hebel(sigma: float, ziel: float, tage: int = T) -> float:
    return math.sqrt(2.0 * math.log(ziel)) / (sigma * math.sqrt(tage))


def simuliere_faktor(sigma: float, hebel: float, gebuehr_pa: float = 0.01,
                     seed: int = 3) -> np.ndarray:
    """Taeglicher Reset, driftfrei, mit Berechnungsgebuehr."""
    rng = np.random.default_rng(seed)
    r = sigma * rng.standard_t(4.0, size=(N, T)) / math.sqrt(4.0 / 2.0)
    tag = np.maximum(1.0 + hebel * r, 0.0) * (1.0 - gebuehr_pa / 252)
    return EINSATZ * np.prod(tag, axis=1)


def kennzahlen(werte: np.ndarray) -> dict:
    return {"Median": np.median(werte), "p90": np.percentile(werte, 90),
            "p99": np.percentile(werte, 99), "p99.9": np.percentile(werte, 99.9),
            "P>=100k": (werte >= 100_000).mean(), "P>=5x": (werte >= 5 * EINSATZ).mean(),
            "P(<20% uebrig)": (werte < 0.2 * EINSATZ).mean()}


def main() -> None:
    print(f"20.000 EUR im Hebel-Sleeve ueber {T} Handelstage, driftfrei, "
          f"{N:,} Pfade\n")
    print("OPTIMALER FAKTOR-HEBEL (geschlossene Loesung, Ziel 5x)")
    print(f"{'Basiswert':<14}{'Tagesvola':>11}{'L* fuer 5x':>12}"
          f"{'bei SG max':>12}{'Bewertung':>28}")
    print("-" * 77)
    for b, s in BASISWERTE.items():
        Lopt = optimaler_hebel(s, 5.0)
        mx = VERFUEGBAR[b]
        if Lopt > mx * 1.3:
            urteil = "SG-Hebel zu niedrig"
        elif Lopt < mx * 0.7:
            urteil = "max. Hebel waere zu hoch"
        else:
            urteil = "Angebot passt zum Optimum"
        print(f"{b:<14}{s*100:>10.2f}%{Lopt:>12.1f}{mx:>12}{urteil:>28}")

    print("\n\nWAS WIRD AUS DEN 20.000 EUR? (Faktor-Zertifikat, Kauf und Halten)")
    print(f"{'Basiswert':<14}{'Hebel':>7}{'Median':>10}{'p90':>10}{'p99':>11}"
          f"{'p99.9':>12}{'P>=100k':>10}{'P<20% uebrig':>14}")
    print("-" * 88)
    besten = {}
    for b, s in BASISWERTE.items():
        kand = sorted({1, 2, 3, 5, 8, 10, 12, VERFUEGBAR[b]})
        best, bestk = None, -1
        for L in kand:
            if L > VERFUEGBAR[b]:
                continue
            k = kennzahlen(simuliere_faktor(s, L))
            if k["P>=100k"] > bestk:
                best, bestk = (L, k), k["P>=100k"]
        L, k = best
        besten[b] = (L, k)
        print(f"{b:<14}{'x'+str(L):>7}{k['Median']:>10,.0f}{k['p90']:>10,.0f}"
              f"{k['p99']:>11,.0f}{k['p99.9']:>12,.0f}{k['P>=100k']:>10.2%}"
              f"{k['P(<20% uebrig)']:>14.1%}")

    print("\n\nEINZELVERGLEICH DAX: Hebel gegen Ergebnis")
    print(f"{'Hebel':>7}{'Median':>10}{'p99':>11}{'p99.9':>12}{'P>=100k':>10}")
    print("-" * 50)
    for L in (3, 5, 8, 10, 12, 15):
        k = kennzahlen(simuliere_faktor(BASISWERTE["DAX 40"], L))
        print(f"{'x'+str(L):>7}{k['Median']:>10,.0f}{k['p99']:>11,.0f}"
              f"{k['p99.9']:>12,.0f}{k['P>=100k']:>10.2%}")

    print("\n\nEINZELVERGLEICH NVIDIA: Hebel gegen Ergebnis")
    print(f"{'Hebel':>7}{'Median':>10}{'p99':>11}{'p99.9':>12}{'P>=100k':>10}")
    print("-" * 50)
    for L in (2, 3, 5, 8, 10, 12, 15):
        k = kennzahlen(simuliere_faktor(BASISWERTE["NVIDIA"], L))
        print(f"{'x'+str(L):>7}{k['Median']:>10,.0f}{k['p99']:>11,.0f}"
              f"{k['p99.9']:>12,.0f}{k['P>=100k']:>10.2%}")

    print("\n\nFAZIT")
    rang = sorted(besten.items(), key=lambda kv: -kv[1][1]["P>=100k"])
    for b, (L, k) in rang:
        print(f"  {b:<14} Faktor x{L:<3} -> P(20k werden 100k) = {k['P>=100k']:.2%}, "
              f"Median {k['Median']:,.0f} EUR")


if __name__ == "__main__":
    main()
