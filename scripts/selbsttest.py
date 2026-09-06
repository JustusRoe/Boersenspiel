#!/usr/bin/env python3
"""Prueft die Zahlen nach, auf denen die Empfehlung beruht.

Kein Ersatz fuer eigenes Nachdenken, aber es trennt das Belastbare vom
Modellabhaengigen: Datenintegritaet gegen unabhaengige Quellen, die
Monte-Carlo-Ergebnisse gegen die analytische Loesung, und die Robustheit
gegenueber den Annahmen, die frei gewaehlt sind.
"""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from strategy import data, rules                                   # noqa: E402

T = 40
EINSATZ = rules.leverage_budget(rules.START_CAPITAL)
ok = fail = 0

def pruefe(name, bedingung, detail=""):
    global ok, fail
    if bedingung:
        ok += 1; print(f"  [ok]   {name} {detail}")
    else:
        fail += 1; print(f"  [FEHL] {name} {detail}")

print("1. REGELN gegen den Regeltext")
print("-" * 74)
pruefe("Hebelkappe bei 100k Depot = 20.000", rules.leverage_budget(100_000) == 20_000)
pruefe("Hebelkappe bei 400k Depot = 20.000 (absolut, nicht 20 %)",
       rules.leverage_budget(400_000) == 20_000, f"-> {rules.leverage_budget(400_000):,.0f}")
pruefe("20.000 Stueck greift bei 0,50 EUR",
       rules.max_units_affordable(0.50, 100_000, True) == 20_000,
       f"-> nur {20_000*0.50:,.0f} EUR investierbar")
pruefe("Aktie unter 1,00 EUR wird abgelehnt",
       not rules.check_buy(100_000, 100_000, 0.80, 100, leveraged=False).ok)
pruefe("40 Handelstage, 8 Spielwochen",
       len(rules.trading_days()) == 40 and len(rules.game_weeks()) == 8)

print("\n2. MONTE CARLO gegen die analytische Loesung")
print("-" * 74)
print("  Median = exp(-(L*sigma)^2*T/2) folgt aus log(1+x) ~ x - x^2/2 und gilt")
print("  nur fuer kleine L*sigma. Geprueft wird deshalb, wo sie traegt --")
print("  nicht ob sie ueberall traegt.")
for sig, L, lab, gueltig in [(0.0056, 15, "DAX x15", True),
                             (0.0287, 5, "NVIDIA x5", True),
                             (0.0287, 10, "NVIDIA x10", False),
                             (0.0287, 15, "NVIDIA x15", False)]:
    x = L * sig
    analytisch = EINSATZ * math.exp(-(x ** 2) * T / 2)
    rng = np.random.default_rng(999)          # anderer Seed als in der Studie
    r = sig * rng.standard_normal((120_000, T))
    sim = np.median(EINSATZ * np.prod(np.maximum(1 + L * r, 0.0), axis=1))
    abw = abs(sim - analytisch) / analytisch
    if gueltig:
        pruefe(f"{lab:<11} L*sigma={x:.3f} analytisch {analytisch:>7,.0f} / "
               f"simuliert {sim:>7,.0f}", abw < 0.05, f"Abw {abw*100:.1f} %")
    else:
        print(f"  [info] {lab:<11} L*sigma={x:.3f} -> Naeherung bricht "
              f"({abw*100:.0f}% Abw), Simulation ist massgeblich")

print("\n  Traegt die Formel wenigstens das OPTIMUM? (dort zaehlt sie)")
for name, sig, maxL in [("DAX 40", 0.0056, 15), ("NVIDIA", 0.0287, 15),
                        ("Tesla", 0.0320, 15)]:
    Lf = math.sqrt(2 * math.log(5)) / (sig * math.sqrt(T))
    best = (None, -1.0)
    for L in range(1, maxL + 1):
        rng = np.random.default_rng(7)
        z = rng.standard_t(4.0, size=(60_000, T)) / math.sqrt(2.0)
        v = EINSATZ * np.prod(np.maximum(1 + L * sig * z, 0.0), axis=1)
        p = float((v >= 100_000).mean())
        if p > best[1]:
            best = (L, p)
    nah = (Lf >= maxL) or abs(Lf - best[0]) <= 2
    pruefe(f"{name:<8} Formel L*={Lf:>5.1f}, simuliert bestes L={best[0]:>2}",
           nah, f"P={best[1]:.2%}")

print("\n3. ROBUSTHEIT: kippt die Rangfolge bei anderen Annahmen?")
print("-" * 74)
def p_ziel(sig, L, df, seed):
    rng = np.random.default_rng(seed)
    z = rng.standard_t(df, size=(80_000, T)) / math.sqrt(df / (df - 2)) if df else rng.standard_normal((80_000, T))
    v = EINSATZ * np.prod(np.maximum(1 + L * sig * z, 0.0), axis=1)
    return (v >= 100_000).mean()
print(f"  {'Annahme':<34}{'DAX x15':>12}{'NVIDIA x10':>14}{'Faktor':>10}")
for lab, df, sd, vmul in [("Basisfall (t, df=4)", 4, 1, 1.0),
                          ("Normalverteilung statt t", None, 2, 1.0),
                          ("t mit df=3 (fettere Raender)", 3, 3, 1.0),
                          ("Vola 30 % niedriger", 4, 4, 0.7),
                          ("Vola 50 % hoeher", 4, 5, 1.5)]:
    a = p_ziel(0.0056 * vmul, 15, df, sd); b = p_ziel(0.0287 * vmul, 10, df, sd)
    print(f"  {lab:<34}{a:>11.2%}{b:>13.2%}{(b/max(a,1e-9)):>9.0f}x")
pruefe("NVIDIA schlaegt DAX unter allen fuenf Annahmen",
       all(p_ziel(0.0287*v, 10, d, s) > p_ziel(0.0056*v, 15, d, s)
           for d, s, v in [(4,1,1.0),(None,2,1.0),(3,3,1.0),(4,4,0.7),(4,5,1.5)]))

print("\n4. DATEN gegen unabhaengige Quellen")
print("-" * 74)
try:
    import pandas as pd
    from strategy.sg_api import fx_to_eur, infer_spot
    for f, asset, sym in [("data/sg/dax.csv", "DAX 40", "^GDAXI"),
                          ("data/sg/einzelaktien.csv", "Tesla", "TSLA"),
                          ("data/sg/einzelaktien.csv", "NVIDIA", "NVDA")]:
        recs = pd.read_csv(f).query("_asset_name == @asset").to_dict("records")
        fx = fx_to_eur(recs); sg = infer_spot(recs, fx)
        yh = data.last_price(sym)
        eur = yh / (data.last_price("EURUSD=X") or 1.16) if sym in ("TSLA", "NVDA") else yh
        abw = abs(sg - eur) / eur
        pruefe(f"{asset:<8} SG {sg:>10,.2f} / Referenz {eur:>10,.2f}", abw < 0.03,
               f"Abweichung {abw*100:.2f} %")
except Exception as e:
    print(f"  [uebersprungen] {type(e).__name__}: {e}")

print("\n" + "=" * 74)
print(f"{ok} Pruefungen bestanden, {fail} fehlgeschlagen")
print("=" * 74)
