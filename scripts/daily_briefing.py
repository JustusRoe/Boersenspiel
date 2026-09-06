#!/usr/bin/env python3
"""Tages-Briefing fuer die 3-5 Checkpoints eines Spieltags.

Aufruf:
    python3 scripts/daily_briefing.py --depot-a 100000 --depot-b 100000
    python3 scripts/daily_briefing.py --depot-a 87000 --depot-b 143000 --no-screen

Gibt aus: Marktlage, anstehende Katalysatoren, Ruecksetzer-Empfehlung,
verbleibende Regelbudgets und die aktuellen Screener-Kandidaten.
Es werden nur Vorschlaege erzeugt -- jede Order gibt der Spieler selbst auf.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy import calendar_events, data, rules      # noqa: E402


def rule_budget(label: str, value: float) -> None:
    lev = rules.leverage_budget(value)
    print(f"  {label}: Depotwert {value:>10,.0f} EUR")
    print(f"      Hebelbudget beim Kauf     {lev:>10,.0f} EUR "
          f"({lev / value * 100:.1f} % des Depots)")
    print(f"      max. je Wertpapier        {rules.max_position_eur(value):>10,.0f} EUR")
    # 5 x 20 % waeren 100 % -- zusammen mit dem Hebelbudget passt das nicht ins
    # Depot. Praktikabel ist: Hebelbudget voll, Rest auf 4-5 Aktien verteilt.
    rest = value - lev
    print(f"      Rest fuer Aktien          {rest:>10,.0f} EUR "
          f"= 4 x {rest / 4:,.0f} oder 5 x {rest / 5:,.0f}")
    print(f"      Stueck-Kappe je Produkt   {rules.MAX_LEVERAGE_UNITS:>10,d} Stueck "
          f"-> ein Schein unter {lev / rules.MAX_LEVERAGE_UNITS:.2f} EUR nimmt das "
          f"Budget nicht auf;")
    print(f"      {'':26s}{'':>10s} Faustregel: mind. 1/Preis Produkte kombinieren")


def reset_advice(label: str, value: float, reset_used_this_week: bool) -> None:
    if value < rules.START_CAPITAL:
        if reset_used_this_week:
            print(f"  {label}: {value:,.0f} EUR < 100.000 -- Ruecksetzer diese Woche "
                  f"bereits verbraucht. Naechster Montag.")
        else:
            gap = rules.START_CAPITAL - value
            print(f"  {label}: RUECKSETZEN. {value:,.0f} EUR liegt {gap:,.0f} EUR unter "
                  f"dem Startwert -- der Ruecksetzer ist geschenktes Kapital.")
    else:
        print(f"  {label}: {value:,.0f} EUR >= 100.000 -- laufen lassen, "
              f"Ruecksetzer fuer die Woche aufheben.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depot-a", type=float, default=100_000.0)
    ap.add_argument("--depot-b", type=float, default=100_000.0)
    ap.add_argument("--reset-a-used", action="store_true")
    ap.add_argument("--reset-b-used", action="store_true")
    ap.add_argument("--no-screen", action="store_true")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    today = date.today()
    print("=" * 78)
    print(f"TRADER 2026 -- BRIEFING {today:%A, %d.%m.%Y}")
    print("=" * 78)

    if today < rules.GAME_START:
        print(f"Spielstart in {(rules.GAME_START - today).days} Tag(en).")
    elif today > rules.GAME_END:
        print("Spiel beendet.")
    else:
        td = [d for d in rules.trading_days() if d >= today]
        print(f"Verbleibende Handelstage: {len(td)}")

    print("\n--- MARKTLAGE " + "-" * 63)
    snap = data.market_snapshot()
    if not snap.empty:
        print(snap.to_string(index=False))
        vix = snap.loc[snap["Markt"] == "VIX", "Kurs"]
        if not vix.empty and vix.iloc[0] is not None:
            v = float(vix.iloc[0])
            if v < 16:
                print("\n  VIX niedrig: Optionsscheine sind billig, Konvexitaet kostet wenig. "
                      "Guenstiges Umfeld, um Hebel zu kaufen statt zu verkaufen.")
            elif v > 25:
                print("\n  VIX erhoeht: Scheine sind teuer, Knock-out-Risiko real. "
                      "Enge Barrieren meiden, Abstand kaufen.")

    print("\n--- KATALYSATOREN (14 Tage) " + "-" * 49)
    ups = calendar_events.upcoming(today, 14)
    if ups:
        for e in ups:
            mark = "***" if e.impact >= 3 else "   "
            print(f"  {mark} {e.day:%a %d.%m.}  {e.label}")
            if e.note:
                print(f"           {e.note}")
    else:
        print("  Keine markierten Termine -- ruhige Phase, Positionen laufen lassen.")

    print("\n--- RUECKSETZER-ENTSCHEIDUNG " + "-" * 48)
    reset_advice("Depot A", args.depot_a, args.reset_a_used)
    reset_advice("Depot B", args.depot_b, args.reset_b_used)

    print("\n--- REGELBUDGETS " + "-" * 60)
    rule_budget("Depot A", args.depot_a)
    rule_budget("Depot B", args.depot_b)
    print(f"\n  Kauforders: max. {rules.MAX_BUY_ORDERS_PER_DAY} je Depot und Tag "
          f"(Verkaeufe unbegrenzt).")
    print(f"  Haltefrist: {rules.HOLDING_PERIOD_MINUTES} Minuten. Verkaufslimits duerfen "
          f"sofort platziert werden -- Take-Profit-Leiter direkt beim Kauf setzen.")

    if not args.no_screen:
        print("\n--- AKTIEN-KANDIDATEN " + "-" * 55)
        from strategy.screener import scan
        df = scan(max_symbols=55)
        if df.empty:
            print("  Screener lieferte keine Daten (Netz/Proxy?).")
        else:
            cols = ["Symbol", "Kurs EUR", "1T %", "5T %", "Vola ann %",
                    "ATR %", "RelVol", "z. 52W-Hoch %", "Score"]
            print(df[cols].head(args.top).to_string(index=False))
            out = Path(__file__).resolve().parent.parent / "data" / "screen_latest.csv"
            df.to_csv(out, index=False)
            print(f"\n  vollstaendig: {out}")

    print("\n" + "=" * 78)
    print("Alle Orders manuell aufgeben. Automatisierte Handelssysteme sind nach")
    print("Ziffer 10 der Spielregeln untersagt und fuehren zum Ausschluss.")
    print("=" * 78)


if __name__ == "__main__":
    main()
