#!/usr/bin/env python3
"""Die strategischen Kernfragen quantitativ beantworten.

  1. Wie aggressiv muss man spielen, um P(Gesamtsieg) zu maximieren?
  2. Was ist die woechentliche Ruecksetzer-Option wert?
  3. Welcher Hebel ist optimal -- und haengt das vom Zielniveau ab?
  4. Was kostet die 20.000-Stueck-Regel, wenn sie je Depot statt je Produkt gilt?
  5. Wie sieht der optimale Einsatz fuer die beste Tagesperformance aus?
  6. Was bringt die Rollenteilung der beiden Depots?

Aufruf:  python3 scripts/run_study.py [--sims 20000] [--quick]
"""

from __future__ import annotations

import argparse
import sys
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy import rules                                        # noqa: E402
from strategy.optimizer import (Turbo, apply_fair_pricing,  # noqa: E402
                                optimise_basket, payoff_per_unit,
                                simulate_underlying_paths, synthetic_universe)
from strategy.simulator import (LeverageSleeve, MarketModel, Policy,  # noqa: E402
                                SimResult, StockSleeve, simulate,
                                simulate_two_depots)

OUT = []


def say(line: str = "") -> None:
    print(line)
    OUT.append(line)


def h(title: str) -> None:
    say(); say("=" * 78); say(title); say("=" * 78)


# ------------------------------------------------------------------ Policies
def build_policies() -> dict[str, Policy]:
    ruhig = StockSleeve(n_positions=5, annual_vol=0.22, beta=1.0,
                        jump_prob=0.001, jump_up=0.15, jump_down=-0.12)
    mittel = StockSleeve(n_positions=5, annual_vol=0.40, beta=1.1,
                         jump_prob=0.006, jump_up=0.35, jump_down=-0.25)
    # Kalibriert auf den Screener vom 06.09.2026: die 25 hoechstbewerteten
    # Kandidaten haben 70-144 % annualisierte 20-Tage-Vola (AXTI 134, AEHR 126,
    # NBIS 144, MSTR 106). jump_up_share bleibt None = erwartungswertneutral.
    wild = StockSleeve(n_positions=5, annual_vol=1.00, beta=1.2,
                       jump_prob=0.020, jump_up=0.70, jump_down=-0.40)
    # Variante mit unterstelltem Selektionsvorteil: 45 % der Spruenge nach oben
    # statt der neutralen 36,4 %. Zeigt, was gutes Stock-Picking wert waere.
    wild_edge = replace(wild, jump_up_share=0.45)

    kein_hebel = LeverageSleeve(n_products=0)
    mild = LeverageSleeve(n_products=2, initial_leverage=6.0, target_price=1.00,
                          underlying_annual_vol=0.16)
    scharf = LeverageSleeve(n_products=4, initial_leverage=25.0, target_price=0.25,
                            underlying_annual_vol=0.16)
    extrem = LeverageSleeve(n_products=4, initial_leverage=60.0, target_price=0.25,
                            underlying_annual_vol=0.16)

    return {
        "1 Buy&Hold Index (kein Hebel, kein Reset)":
            Policy("buyhold", ruhig, kein_hebel, use_resets=False),
        "2 Solide Aktien + milder Hebel, mit Reset":
            Policy("solide", mittel, mild),
        "3 Hochvola-Aktien + scharfer Hebel, OHNE Reset":
            Policy("scharf_ohne_reset", wild, scharf, use_resets=False),
        "4 Hochvola-Aktien + scharfer Hebel, MIT Reset":
            Policy("scharf", wild, scharf),
        "5 Maximale Konvexitaet (x60) MIT Reset":
            Policy("extrem", wild, extrem),
        "6 wie 5, aber mit Stock-Picking-Vorteil":
            Policy("extrem_edge", wild_edge, extrem),
    }


def fmt(res: SimResult) -> str:
    s = res.summary()
    return (f"{s['median']:>10,.0f} {s['p90']:>11,.0f} {s['p99']:>11,.0f} "
            f"{s['p99.9']:>11,.0f} {s['P(>=250k)']:>9.2%} {s['P(>=500k)']:>9.2%} "
            f"{s['P(<100k)']:>8.1%} {s['best_day_p99']:>9.1%} {s['resets_mean']:>7.1f}")


def study_policies(n_sims: int) -> dict[str, SimResult]:
    h("STUDIE 1 -- Wie aggressiv muss man spielen? (je Depot, 40 Handelstage)")
    say(f"{'Strategie':<44}{'Median':>10}{'p90':>11}{'p99':>11}"
        f"{'p99.9':>11}{'P>=250k':>9}{'P>=500k':>9}{'P<100k':>8}{'BestTag':>9}{'Resets':>7}")
    say("-" * 129)
    out = {}
    for label, pol in build_policies().items():
        # deterministischer Seed je Strategie (hash() waere prozessabhaengig)
        r = simulate(pol, n_sims=n_sims, seed=zlib.crc32(label.encode()) % 9000)
        out[label] = r
        say(f"{label:<44}{fmt(r)}")
    say()
    say("Lesart: Der Median faellt mit steigender Aggressivitaet, aber nur die")
    say("rechte Verteilungsschulter (p99.9 / P>=250k) entscheidet ueber den Sieg.")
    say("Eine Strategie mit hoeherem Median und niedrigerem p99.9 ist im Spiel")
    say("strikt schlechter -- es gibt keinen Preis fuer einen guten Median.")
    return out


def study_reset(n_sims: int) -> None:
    h("STUDIE 2 -- Was ist die woechentliche Ruecksetzer-Option wert?")
    base = build_policies()["5 Maximale Konvexitaet (x60) MIT Reset"]
    say(f"{'Variante':<44}{'Median':>10}{'p99':>11}{'p99.9':>11}"
        f"{'P>=250k':>10}{'P<100k':>9}")
    say("-" * 95)
    for label, pol in [
        ("ohne Ruecksetzer", replace(base, use_resets=False)),
        ("Reset bei < 100.000 (rational)", replace(base, use_resets=True, reset_below=100_000)),
        ("Reset bei < 80.000 (zoegerlich)", replace(base, use_resets=True, reset_below=80_000)),
        ("Reset bei < 130.000 (zu eifrig)", replace(base, use_resets=True, reset_below=130_000)),
    ]:
        s = simulate(pol, n_sims=n_sims, seed=4242).summary()
        say(f"{label:<44}{s['median']:>10,.0f}{s['p99']:>11,.0f}{s['p99.9']:>11,.0f}"
            f"{s['P(>=250k)']:>10.2%}{s['P(<100k)']:>9.1%}")
    say()
    say("Der Ruecksetzer schneidet den linken Verteilungsrand ab, ohne den")
    say("rechten zu beschneiden. Genau deshalb ist maximales Risiko rational:")
    say("Der Verlust ist auf 0 % begrenzt, nicht auf -100 %.")
    say("Eine zu eifrige Schwelle (>100k) bringt bei P(>=250k) nichts mehr,")
    say("kostet aber deutlich Median: man wirft Substanz weg, die man schon")
    say("hatte. 100.000 EUR ist genau die richtige Schwelle -- exakt der Punkt,")
    say("ab dem der Ruecksetzer geschenktes Kapital statt Verlust ist.")


def study_leverage(n_sims: int) -> None:
    h("STUDIE 3 -- Welcher Hebel ist optimal?")
    base = build_policies()["5 Maximale Konvexitaet (x60) MIT Reset"]
    say(f"{'Turbo-Hebel':<44}{'Median':>10}{'p99':>11}{'p99.9':>11}{'P>=250k':>10}{'P>=500k':>10}")
    say("-" * 95)
    for L in (4, 8, 15, 25, 40, 60, 90, 130):
        pol = replace(base, leverage=replace(base.leverage, initial_leverage=float(L)))
        s = simulate(pol, n_sims=n_sims, seed=99).summary()
        say(f"{'x' + str(L):<44}{s['median']:>10,.0f}{s['p99']:>11,.0f}"
            f"{s['p99.9']:>11,.0f}{s['P(>=250k)']:>10.2%}{s['P(>=500k)']:>10.2%}")
    say()
    say("Die Turbos sind hier arbitragefrei bepreist: nach Finanzierungskosten")
    say("und eingepreistem Gap-Risiko ist ihr Erwartungswert exakt null.")
    say("Trotzdem steigt mit dem Hebel nicht nur die rechte Schulter, sondern")
    say("auch der Median -- und das ist kein Fehler, sondern der Ruecksetzer:")
    say("die woechentliche Untergrenze bei 100.000 EUR kappt den linken Rand,")
    say("und unter einer solchen Untergrenze verbessert mehr Varianz jedes")
    say("Quantil oberhalb davon. Mit Ruecksetzer ist mehr Risiko schlicht")
    say("dominant. Die Grenze setzt nicht die Mathematik, sondern der Markt:")
    say("Turbos mit Hebel > ~100 haben Spreads und Gap-Aufschlaege, die real")
    say("weit ueber dem hier modellierten Prozent liegen.")


def study_units_rule() -> None:
    h("STUDIE 4 -- Was kostet die 20.000-Stueck-Regel?")
    spot = 26046.4
    say("Frage: Gilt die Stueck-Kappe je Wertpapier oder je Depot? Das ist am")
    say("ersten Spieltag mit zwei Kleinstorders zu klaeren.")
    say()
    say("Der Effekt zeigt sich nur, wenn die attraktivsten Scheine BILLIG sind --")
    say("dann verhindert die Stueck-Kappe, dass die 20.000 EUR investiert werden.")
    say("Deshalb zwei Universen: eines mit Scheinen ab 1,00 EUR, eines nur mit")
    say("Billigscheinen zu 0,10-0,25 EUR.")
    say()
    universen = {
        "Scheine ab 1,00 EUR verfuegbar": (0.50, 1.00, 2.00),
        "nur Billigscheine 0,10-0,25 EUR": (0.10, 0.15, 0.25),
    }
    say(f"{'Universum':<34}{'Auslegung':<26}{'Einsatz EUR':>13}{'P(>=1.5x/Wo)':>15}")
    say("-" * 88)
    for uni_label, prices in universen.items():
        uni = (synthetic_universe("DAX", spot, +1, target_prices=prices)
               + synthetic_universe("DAX", spot, -1, target_prices=prices))
        uni = apply_fair_pricing(uni, {"DAX": 0.16}, 5)
        for cap_label, per_product in [("je Wertpapier", True), ("je Depot", False)]:
            b = optimise_basket(uni, 100_000, 5, {"DAX": 0.16}, "p_target", 1.5,
                                n_paths=20_000, n_restarts=200,
                                units_cap_is_per_product=per_product, seed=3)
            say(f"{uni_label:<34}{cap_label:<26}{b.cost:>13,.0f}{b.objective_value:>14.2%}")
    say()
    say("Der Preis je Schein aendert die Konvexitaet pro Euro NICHT -- das")
    say("Bezugsverhaeltnis gleicht das aus. Die Stueck-Regel kostet nur dann")
    say("etwas, wenn sie verhindert, die vollen 20.000 EUR einzusetzen.")
    say("Praktische Regel: maximaler Hebel, und mindestens 1/Preis Produkte")
    say("kombinieren, damit das Budget voll investiert ist.")


def study_best_day() -> None:
    h("STUDIE 5 -- Einsatz fuer die beste Tagesperformance (1 Handelstag)")
    spot = 26046.4
    for vol, label in [(0.16, "normaler Tag"), (0.30, "Ereignistag (FOMC/EZB)")]:
        uni = synthetic_universe("DAX", spot, +1) + synthetic_universe("DAX", spot, -1)
        uni = apply_fair_pricing(uni, {"DAX": vol}, 1)
        b = optimise_basket(uni, 100_000, 1, {"DAX": vol}, "p_target", 1.30,
                            n_paths=30_000, n_restarts=250, seed=5)
        say()
        say(f"--- {label} (Basiswert-Vola {vol:.0%}) ---")
        say(b.to_table())
        for k, v in b.stats.items():
            say(f"   {k:30s} {v:,.4f}")
    say()
    say("Wichtig: Das Hebelbudget ist absolut auf 20.000 EUR gedeckelt. Bei")
    say("einem 100k-Depot sind das 20 % Wirkung, bei einem 300k-Depot nur 6,7 %.")
    say("Der Tagesperformance-Preis ist damit strukturell an ein frisch")
    say("zurueckgesetztes 100k-Depot gebunden.")


def study_two_depots(n_sims: int) -> None:
    h("STUDIE 6 -- Rollenteilung der beiden Depots")
    pols = build_policies()
    compounder = pols["4 Hochvola-Aktien + scharfer Hebel, MIT Reset"]  # noqa
    sniper = replace(pols["5 Maximale Konvexitaet (x60) MIT Reset"],
                     reset_below=100_000)

    combos = [
        ("beide identisch (scharf)", compounder, compounder),
        ("Compounder + Sniper", compounder, sniper),
        ("beide maximal (x60)", sniper, sniper),
    ]
    say(f"{'Aufstellung':<34}{'P(bestes>=250k)':>18}{'P(bestes>=500k)':>18}"
        f"{'bester Tag p99':>17}{'beste Woche p99':>18}")
    say("-" * 105)
    for label, a, b in combos:
        r = simulate_two_depots(a, b, n_sims=n_sims, seed=17)
        say(f"{label:<34}{r['P(best>=250k)']:>18.2%}{r['P(best>=500k)']:>18.2%}"
            f"{np.percentile(r['best_day_overall'], 99):>16.1%}"
            f"{np.percentile(r['best_week_overall'], 99):>17.1%}")
    say()
    say("Zwei Depots sind zwei unabhaengige Lose. Fuer den Gesamtsieg zaehlt nur")
    say("das bessere, also senkt Diversifikation zwischen den Depots die")
    say("Siegchance -- sie erhoeht sie nicht.")


def study_turbo_vs_factor() -> None:
    h("STUDIE 7 -- Knock-out-Turbo oder Faktor-Zertifikat fuer die 20.000 EUR?")
    say("Der entscheidende Unterschied liegt nicht in der Rendite, sondern im")
    say("Slot: Das Hebelbudget ist dauerhaft auf 20.000 EUR gedeckelt. Knockt")
    say("ein Turbo aus, sind diese 20.000 EUR weg und man darf nur wieder")
    say("20.000 EUR nachlegen -- man kommt nie voran. Ein Faktor-Zertifikat")
    say("kann nicht ausgeknockt werden und zinst denselben Slot ueber acht")
    say("Wochen auf. Dafuer frisst der taegliche Reset in Seitwaertsphasen.")
    say()
    spot = 26046.4
    for horizon, hlabel in [(5, "1 Woche"), (40, "volles Spiel (40 Tage)")]:
        say(f"--- Horizont: {hlabel}, DAX-Vola 16 %, 20.000 EUR Einsatz ---")
        say(f"{'Produkt':<26}{'Median':>10}{'p95':>9}{'p99':>9}{'p99.9':>10}"
            f"{'P(Totalverlust)':>17}")
        say("-" * 81)
        paths = simulate_underlying_paths(spot, 0.16, horizon, 60_000, seed=21)
        S_end, S_min, S_max = paths[:, -1], paths.min(axis=1), paths.max(axis=1)
        rows = []
        for L in (5, 10, 15):
            fac = Turbo(f"FAC{L}", f"Faktor {L}x Long", "DAX", +1, 1.0, 0.99,
                        spot, 0.0, 1.0, float(L), product_type="factor")
            rows.append((f"Faktor-Zertifikat {L}x", payoff_per_unit(fac, S_end, S_min, S_max, paths)))
        for L in (25, 70):
            uni = apply_fair_pricing(
                synthetic_universe("DAX", spot, +1, leverages=(L,), target_prices=(1.0,)),
                {"DAX": 0.16}, horizon)
            t = uni[0]
            rows.append((f"KO-Turbo x{L}", payoff_per_unit(t, S_end, S_min, S_max) / t.ask))
        for label, pay in rows:
            v = 20_000 * pay
            say(f"{label:<26}{np.median(v):>10,.0f}{np.percentile(v, 95):>9,.0f}"
                f"{np.percentile(v, 99):>9,.0f}{np.percentile(v, 99.9):>10,.0f}"
                f"{np.mean(pay <= 0.01):>16.1%}")
        say()
    say("Lesart -- vorsichtig, die Zahlen sind eindeutiger als die uebliche")
    say("Faustregel: Der scharfe Turbo behaelt auch ueber 40 Tage die groesste")
    say("rechte Schulter (p99.9 rund 270k gegen 238k). Er erkauft sie aber mit")
    say("79,6 % Totalverlustwahrscheinlichkeit gegen 0,2 %. Das Faktor-Zertifikat")
    say("holt also rund 88 % des Extremszenarios bei praktisch keinem Ausfall.")
    say()
    say("Wichtige Einschraenkung: Dieser Vergleich haelt eine einzige Position")
    say("40 Tage durch. In der Praxis kann man den Turbo woechentlich neu")
    say("aufsetzen -- dann kostet jeder Knock-out aber erneut 20 % des Depots.")
    say("Genau diesen Effekt misst Studie 3 im vollstaendigen Spielverlauf.")
    say()
    say("Praktische Konsequenz: Faktor-Zertifikat als dauerhafte Hebelbasis im")
    say("Compounder-Depot (der Slot ueberlebt), scharfe Turbos als terminierte")
    say("Ereigniswette im Sniper-Depot (dort ist der Ruecksetzer das Netz).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=20_000)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n = 4_000 if args.quick else args.sims

    say(f"Trader 2026 -- Strategiestudie")
    say(f"Spielzeitraum {rules.GAME_START:%d.%m.%Y} bis {rules.GAME_END:%d.%m.%Y} "
        f"({len(rules.trading_days())} Handelstage, {len(rules.game_weeks())} Wochen)")
    say(f"Monte-Carlo mit {n:,} Pfaden je Variante")

    study_policies(n)
    study_reset(n)
    study_leverage(n)
    study_units_rule()
    study_best_day()
    study_two_depots(n)
    study_turbo_vs_factor()

    path = Path(__file__).resolve().parent.parent / "data" / "study_report.txt"
    path.write_text("\n".join(OUT) + "\n", encoding="utf-8")
    print(f"\n[gespeichert] {path}")


if __name__ == "__main__":
    main()
