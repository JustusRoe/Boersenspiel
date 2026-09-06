#!/usr/bin/env python3
"""Erzeugt die konkrete Orderliste fuer einen Spieltag.

Prueft jede Order gegen die Spielregeln (20 %, 20.000 EUR, 20.000 Stueck,
20 Kauforders, Mindestkurs 1,00 EUR) und beruecksichtigt, an welchen
Referenzmaerkten ueberhaupt gehandelt wird.

    python3 scripts/orders_today.py                      # aus Cache
    python3 scripts/orders_today.py --refresh            # frische SG-Kurse
    python3 scripts/orders_today.py --depot-a 87000 --depot-b 143000
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy import rules                                          # noqa: E402
from strategy.sg_api import (ASSETS, CLS_BEST_TURBO_OPEN_END,        # noqa: E402
                             CLS_FAKTOR_OS_LEAF, CLS_UNLIMITED_TURBO_MINI,
                             SGClient, enrich_factors, fx_to_eur,
                             infer_spot, to_turbos, validate_pricing)


def _nth_weekday(y: int, m: int, wd: int, n: int) -> date:
    d = date(y, m, 1)
    while d.weekday() != wd:
        d += timedelta(days=1)
    return d + timedelta(days=7 * (n - 1))


def us_feiertage(jahr: int) -> dict[date, str]:
    """US-Boersenfeiertage, die in den Spielzeitraum fallen koennen.

    Wichtig fuer den Spielstart: Der erste Montag im September ist Labor Day.
    Die Boerse Stuttgart handelt zwar, aber ohne Referenzmarkt -- US-Aktien
    und Scheine auf US-Basiswerte werden dann nur indikativ und mit weiten
    Spreads gestellt. Columbus Day betrifft nur den Anleihemarkt.
    """
    return {_nth_weekday(jahr, 9, 0, 1): "Labor Day (US-Aktienmarkt geschlossen)",
            _nth_weekday(jahr, 11, 3, 4) + timedelta(days=1): "Thanksgiving-Freitag (verkuerzt)"}


def lade_dax(refresh: bool, cache: Path) -> tuple[float, list]:
    if refresh or not cache.exists():
        c = SGClient()
        recs = []
        for cls in (CLS_BEST_TURBO_OPEN_END, CLS_UNLIMITED_TURBO_MINI):
            recs += list(c.iter_products(cls, ASSETS["DAX 40"], page_size=1000,
                                         max_products=2500, verbose=False))
        fac = list(c.iter_products(CLS_FAKTOR_OS_LEAF, ASSETS["DAX 40"],
                                   page_size=1000, verbose=False))
        enrich_factors(c, fac, workers=6, verbose=False)
        recs += fac
        df = pd.DataFrame(recs).drop_duplicates(subset=["Code"])
        df["_asset_name"] = "DAX 40"
        df["_spot"] = infer_spot(recs, fx_to_eur(recs))
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
    df = pd.read_csv(cache)
    recs = df.to_dict("records")
    spot = infer_spot(recs, fx_to_eur(recs))
    ts = to_turbos(recs, spot=spot, verbose=False)
    ts, _ = validate_pricing(ts, spot, verbose=False)
    return spot, ts


def order_zeile(nr: int, depot: str, was: str, wkn: str, stueck: int,
                brief: float, hinweis: str = "") -> str:
    kosten = stueck * brief
    return (f"  {nr}. [{depot}] {was:<34} WKN {wkn:<8} {stueck:>7,} Stueck "
            f"a {brief:>7.3f} = {kosten:>8,.0f} EUR   {hinweis}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depot-a", type=float, default=100_000.0)
    ap.add_argument("--depot-b", type=float, default=100_000.0)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--cache", default="data/sg/dax_aktuell.csv")
    ap.add_argument("--tag", help="ISO-Datum, sonst heute")
    args = ap.parse_args()

    heute = date.fromisoformat(args.tag) if args.tag else date.today()
    print("=" * 84)
    print(f"ORDERLISTE  {heute:%A, %d.%m.%Y}")
    print("=" * 84)

    feiertag = us_feiertage(heute.year).get(heute)
    us_offen = feiertag is None
    if feiertag:
        print(f"\n  ACHTUNG: {feiertag}")
        print("  Die Boerse Stuttgart handelt, aber US-Basiswerte haben heute keinen")
        print("  Referenzmarkt. Kurse sind indikativ, Spreads weit. Heute deshalb nur")
        print("  europaeische Basiswerte handeln, den Aktien-Sleeve auf morgen 15:30")
        print("  verschieben. Ein Tag weniger Exposure kostet fast nichts -- ein")
        print("  ganzes Spiel mit schwachvolatilen Titeln kostet viel.")

    spot, ts = lade_dax(args.refresh, Path(args.cache))
    print(f"\n  DAX (aus Produktdaten abgeleitet): {spot:,.2f}")
    print(f"  Datenstand: {'frisch abgerufen' if args.refresh else Path(args.cache).name}")

    budget_a = rules.leverage_budget(args.depot_a)
    budget_b = rules.leverage_budget(args.depot_b)

    def waehle_faktor(direction: int, budget: float):
        k = [t for t in ts if t.product_type == "factor" and t.direction == direction
             and t.leverage and abs(t.leverage) >= 10 and t.spread_pct <= 0.005
             and min(int(budget // t.ask), rules.MAX_LEVERAGE_UNITS) * t.ask >= 0.97 * budget]
        return max(k, key=lambda t: (abs(t.leverage), -t.spread_pct)) if k else None

    def waehle_turbo(direction: int, budget: float, ko_min: float, ko_max: float):
        k = [t for t in ts if t.product_type == "turbo" and t.direction == direction
             and ko_min <= t.distance_to_ko_pct <= ko_max and t.spread_pct <= 0.005
             and min(int(budget // t.ask), rules.MAX_LEVERAGE_UNITS) * t.ask >= 0.97 * budget]
        return min(k, key=lambda t: t.spread_pct) if k else None

    fa = waehle_faktor(+1, budget_a)

    # Deckt der Ruecksetzer den Hebel-Sleeve noch ab?
    #
    # Der Ruecksetzer stellt 100.000 EUR her. Faellt der Sleeve komplett aus,
    # landet das Depot bei (Depotwert - Einsatz). Liegt dieser Rest UNTER
    # 100.000, holt der Ruecksetzer am Montag den vollen Verlust zurueck -- die
    # Fehlwette kostet dann effektiv nichts, und maximale Konvexitaet ist
    # dominant. Liegt der Rest darueber, waere ein Ruecksetzer Wertvernichtung,
    # der Ausfall also ein echter Verlust. Ab da lohnen weitere Barrieren und
    # Absicherungen. Die Schwelle liegt bei rund 120.000 EUR Depotwert.
    reset_deckt_b = (args.depot_b - budget_b) < rules.START_CAPITAL
    if reset_deckt_b:
        tb = (waehle_turbo(-1, budget_b, 0.008, 0.020)
              or waehle_turbo(-1, budget_b, 0.005, 0.035))
    else:
        tb = (waehle_turbo(-1, budget_b, 0.030, 0.060)
              or waehle_turbo(-1, budget_b, 0.020, 0.090))

    print("\n" + "-" * 84)
    print("HEUTE AUFGEBEN (europaeische Basiswerte, Xetra offen ab 9:00)")
    print("-" * 84)
    nr = 0
    if fa:
        nr += 1
        st = min(int(budget_a // fa.ask), rules.MAX_LEVERAGE_UNITS)
        chk = rules.check_buy(args.depot_a, args.depot_a, fa.ask, st, leveraged=True)
        print(order_zeile(nr, "A", f"Faktor-Zertifikat DAX +{abs(fa.leverage):.0f}x",
                          fa.wkn, chk.units, fa.ask,
                          f"Spread {fa.spread_pct*100:.2f} %"))
        print(f"      Dauerposition fuer acht Wochen. Kein Knock-out -- der Hebel-Slot")
        print(f"      ueberlebt, deshalb hier ein Faktor-Zertifikat statt eines Turbos.")
    if tb:
        nr += 1
        st = min(int(budget_b // tb.ask), rules.MAX_LEVERAGE_UNITS)
        chk = rules.check_buy(args.depot_b, args.depot_b, tb.ask, st, leveraged=True)
        print(order_zeile(nr, "B", f"DAX Short-Turbo Hebel {tb.omega:.0f}",
                          tb.wkn, chk.units, tb.ask,
                          f"KO {tb.ko_barrier:,.0f} ({tb.distance_to_ko_pct*100:.2f} % entfernt)"))
        print(f"      Gegenrichtung zu Depot A. Der Tagesperformance-Preis ist ein")
        print(f"      Maximum ueber beide Depots, nicht eine Summe -- entgegengesetzte")
        print(f"      Richtungen erhoehen die Trefferwahrscheinlichkeit.")
        if reset_deckt_b:
            print(f"      Enge Barriere gewaehlt: Bei einem Ausfall landet Depot B bei")
            print(f"      {args.depot_b - budget_b:,.0f} EUR, der Ruecksetzer stellt 100.000 her.")
            print(f"      Der Verlust ist also erstattet -- maximale Konvexitaet ist hier")
            print(f"      nicht waghalsig, sondern die richtige Wahl.")
        else:
            print(f"      Weitere Barriere gewaehlt: Bei einem Ausfall bliebe Depot B bei")
            print(f"      {args.depot_b - budget_b:,.0f} EUR, also UEBER 100.000. Ein Ruecksetzer")
            print(f"      waere jetzt Wertvernichtung, der Ausfall damit ein echter Verlust.")

    print("\n" + "-" * 84)
    print("MORGEN AB 15:30 (US-Eroeffnung, Aktien-Sleeve)" if not us_offen
          else "HEUTE AB 15:30 (US-Eroeffnung, Aktien-Sleeve)")
    print("-" * 84)
    try:
        scr = pd.read_csv("data/screen_latest.csv")
        top = scr.head(6)
        rest_a = args.depot_a - budget_a
        print(f"  Je Depot 4 Positionen a {rest_a/4:,.0f} EUR "
              f"(20-%%-Grenze: {rules.max_position_eur(args.depot_a):,.0f} EUR)")
        print(f"  {'Titel':<9}{'Kurs EUR':>10}{'Vola ann':>10}{'ATR':>8}{'RelVol':>8}"
              f"{'Tail p95':>10}{'Score':>8}")
        for _, r in top.iterrows():
            print(f"  {r['Symbol']:<9}{r['Kurs EUR']:>10,.2f}{r['Vola ann %']:>9.0f}%"
                  f"{r['ATR %']:>7.1f}%{r['RelVol']:>8.2f}{r['Tail p95 %']:>9.1f}%"
                  f"{r['Score']:>8.1f}")
        print("\n  Vor dem Kauf pruefen: Ist der Titel in Stuttgart gelistet und steht")
        print("  der Kurs bei mindestens 1,00 EUR? Beides ist Regelvoraussetzung.")
    except FileNotFoundError:
        print("  data/screen_latest.csv fehlt -- scripts/daily_briefing.py ausfuehren.")

    print("\n" + "-" * 84)
    print("REGELBUDGET NACH DIESEN ORDERS")
    print("-" * 84)
    for lab, wert, budget, gewaehlt in (("A", args.depot_a, budget_a, fa),
                                        ("B", args.depot_b, budget_b, tb)):
        genutzt = (min(int(budget // gewaehlt.ask), rules.MAX_LEVERAGE_UNITS)
                   * gewaehlt.ask) if gewaehlt else 0.0
        print(f"  Depot {lab}: Hebelbudget {budget:>8,.0f} EUR, davon genutzt "
              f"{genutzt:>8,.0f} EUR ({genutzt/budget*100:>5.1f} %)")
        print(f"            frei fuer Aktien: {wert - genutzt:>8,.0f} EUR, "
              f"max. je Titel {rules.max_position_eur(wert):,.0f} EUR")
    print(f"  Kauforders heute: {nr} von {rules.MAX_BUY_ORDERS_PER_DAY} je Depot")

    print("\n" + "-" * 84)
    print("AUSFUEHRUNGSHINWEISE")
    print("-" * 84)
    print("  - Nicht um 8:00 handeln. Vor Xetra-Eroeffnung um 9:00 stellt der")
    print("    Market Maker nur indikative Kurse mit weiten Spannen.")
    print("  - Die 5-Minuten-Haltefrist blockiert die Ausfuehrung, nicht die")
    print("    Platzierung: Verkaufslimits koennen sofort beim Kauf hinterlegt")
    print("    werden. Wo das sinnvoll ist, steht im naechsten Punkt.")
    print("  - Stop-Loss und Take-Profit haengen am Depotstand, nicht am Gefuehl:")
    print(f"      Solange (Depotwert - Hebeleinsatz) unter 100.000 liegt -- also bis")
    print(f"      etwa 120.000 EUR Depotwert -- ist ein Stop nutzlos: Was er rettet,")
    print(f"      haette der Ruecksetzer ohnehin erstattet, und er kappt die")
    print(f"      Konvexitaet, die wir gerade bezahlt haben.")
    print(f"      Darueber greift der Ruecksetzer nicht mehr. Dann Stop-Loss auf den")
    print(f"      Hebel-Sleeve setzen, damit erarbeitete Substanz nicht verfaellt.")
    print("  - Take-Profit ist eine Frage des Ziels, nicht der Vorsicht:")
    print("      Depot A (Gesamtsieg) wird zu Marktkursen bewertet, Gewinne muessen")
    print("      nicht realisiert werden -- also halten, nichts verkaufen. Wer einen")
    print("      Gewinner-Turbo verkauft, kann diese Exposure nie wieder aufbauen,")
    print("      weil Neukaeufe fuer immer bei 20.000 EUR gedeckelt sind.")
    print("      Depot B (Tages- und Wochenwertung) wird zum Schlusskurs bewertet.")
    print("      Ein Ausschlag um 16:00, der bis 22:00 zuruecklaeuft, bringt dort")
    print("      nichts -- ausser man hat ihn verkauft. Nur hier lohnt eine")
    print("      Take-Profit-Leiter.")
    print("  - Jede Order von Hand aufgeben. Automatisierte Handelssysteme sind")
    print("    nach Ziffer 10 der Spielregeln untersagt.")


if __name__ == "__main__":
    main()
