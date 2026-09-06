"""Strategie-Werkzeuge fuer das Boersenspiel Trader 2026 der Societe Generale.

Module:
    rules            -- Spielregeln als ausfuehrbare Nebenbedingungen
    simulator        -- Monte-Carlo des gesamten Spiels inkl. Ruecksetzer-Option
    optimizer        -- Konvexitaetsoptimierung des Hebel-Sleeves unter den Kappen
    data             -- Marktdaten (Yahoo) ueber requests
    screener         -- Kandidatensuche fuer den Aktien-Sleeve
    calendar_events  -- Katalysatorkalender des Spielzeitraums
"""

__all__ = ["rules", "simulator", "optimizer", "data", "screener", "calendar_events"]
