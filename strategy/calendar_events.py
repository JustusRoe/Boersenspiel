"""Katalysatorkalender fuer den Spielzeitraum 07.09. - 30.10.2026.

Konvexe Wetten brauchen einen Termin. Diese Liste ist die Grundlage dafuer,
an welchen Tagen der Sniper-Depot-Einsatz platziert wird -- und an welchen
man besser flach bleibt.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from . import rules


@dataclass
class Event:
    day: date
    label: str
    kind: str           # macro | earnings | technical
    impact: int         # 1 = beachten, 2 = relevant, 3 = Sniper-Kandidat
    note: str = ""


EVENTS: list[Event] = [
    Event(date(2026, 9, 10), "EZB-Zinsentscheid 14:15 + PK 14:45", "macro", 3,
          "Voll in der Handelszeit. DAX/Euro-Vola, Bund-Turbos."),
    Event(date(2026, 9, 11), "US CPI (erwartet)", "macro", 2,
          "14:30 MEZ. Termin vor Spielstart gegenpruefen."),
    Event(date(2026, 9, 16), "FOMC-Entscheid 20:00 + Dot Plot + Powell 20:30", "macro", 2,
          "Liegt in der Handelszeit bis 22:00. ABER: empirisch bewegen sich Indizes "
          "an FOMC-Tagen WENIGER als an normalen Tagen (S&P Faktor 0,80, DAX 0,50 "
          "ueber 13 Sitzungen). Kein Tagesperformance-Kandidat, siehe "
          "scripts/fomc_calibration.py."),
    Event(date(2026, 9, 18), "Grosser Verfallstag (Triple Witching)", "technical", 3,
          "Hoechste Umsaetze des Quartals, Pinning und danach Gamma-Unwind."),
    Event(date(2026, 9, 21), "Index-Rebalancing wirksam (Woche)", "technical", 2,
          "STOXX/DAX-Anpassungen, mechanische Flows in Nebenwerten."),
    Event(date(2026, 9, 30), "Quartalsende / Window Dressing", "technical", 2,
          "Umschichtungen, oft Momentum-Verlaengerung in den Quartalsgewinnern."),
    Event(date(2026, 10, 2), "US-Arbeitsmarktbericht (erwartet)", "macro", 2, "14:30 MEZ."),
    Event(date(2026, 10, 13), "Start Q3-Berichtssaison US-Banken", "earnings", 3,
          "Ab hier taeglich Einzelwert-Katalysatoren fuer den Aktien-Sleeve."),
    Event(date(2026, 10, 15), "US-Grossbanken Q3 (u.a. USB 15:00 MEZ)", "earnings", 3, ""),
    Event(date(2026, 10, 20), "Big-Tech-Woche beginnt", "earnings", 3,
          "Zahlen kommen nach US-Schluss (22:05 MEZ), die EUWAX schliesst 22:00. "
          "Die Position muss also ueber die Veroeffentlichung gehalten werden, "
          "die Tagesperformance faellt am Folgetag an."),
    Event(date(2026, 10, 22), "Tesla Q3 (Termin ~21.-28.10., unbestaetigt)", "earnings", 3,
          "Bester Tagesperformance-Kandidat: Tesla-Turbos erreichen eine "
          "Sleeve-Tagesvola von 153 % gegen 34 % bei DAX-Turbos."),
    Event(date(2026, 10, 30), "Coinbase Q3 wirkt (Bericht 29.10. nach Schluss)", "earnings", 3,
          "Faellt exakt auf den letzten Spieltag -- zugleich Tagesperformance-"
          "Chance und Schlussbewertung. Entsprechend riskant fuer den Gesamtrang."),
    Event(date(2026, 10, 28), "FOMC-Entscheid 19:00/20:00", "macro", 3,
          "Zwei Handelstage vor Spielende. Letzte grosse Konvexitaetschance."),
    Event(date(2026, 10, 29), "EZB-Zinsentscheid", "macro", 2, ""),
    Event(date(2026, 10, 30), "LETZTER SPIELTAG", "technical", 3,
          "Depotwert 22:00 zaehlt. Keine Absicherung mehr noetig -- der Endwert ist alles."),
]


def upcoming(today: date | None = None, horizon_days: int = 14) -> list[Event]:
    from datetime import date as _d, timedelta
    today = today or _d.today()
    return [e for e in EVENTS if today <= e.day <= today + timedelta(days=horizon_days)]


def weekly_plan() -> str:
    """Wochenraster: Wann wird zurueckgesetzt, wann eingesetzt, wann gewertet."""
    lines = ["Spielwochen und Taktung", "=" * 72]
    for i, (mon, sun) in enumerate(rules.game_weeks(), start=1):
        evs = [e for e in EVENTS if mon <= e.day <= sun and e.impact >= 3]
        tag = "; ".join(f"{e.day.strftime('%d.%m.')} {e.label}" for e in evs) or "-"
        lines.append(f"KW{i} {mon.strftime('%d.%m.')}-{sun.strftime('%d.%m.')}  "
                     f"Ruecksetzer-Fenster Mo frueh | Wertung So-Nacht")
        lines.append(f"      Katalysatoren: {tag}")
    return "\n".join(lines)
