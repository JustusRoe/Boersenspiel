#!/usr/bin/env python3
"""Lohnt es sich, brachliegendes Cash in Discount-Zertifikaten zu parken?

Das Spiel verzinst Barbestaende nicht -- Ziffer 9 zaehlt schlicht "den
Barbestand" zum Depotwert. Wer Cash haelt, verschenkt also den Geldmarktzins.

Einfangen laesst er sich ueber tief im Geld liegende Discount-Zertifikate:
Liegt der Cap deutlich unter dem Basiswertkurs, ist der eingebaute Short-Call
so gut wie sicher ausgeuebt und das Papier verhaelt sich wie eine
Nullkuponanleihe auf den Cap. Anleihen selbst sind nach Ziffer 4 wegen der
Stueckzinsen ausgeschlossen, Discount-Zertifikate nicht.

Dagegen stehen drei Kosten, von denen die zweite ueberrascht:
  1. 3,90 EUR je Ausfuehrung, also Kauf und Verkauf.
  2. Die 20-%-Regel je Wertpapier erzwingt bei 100.000 EUR Depot mindestens
     fuenf verschiedene Papiere -- und damit zehn Orders statt zwei.
  3. Der Geld-Brief-Spread, mit rund 0,004 % allerdings vernachlaessigbar.

    python3 scripts/cash_parken.py --cash 200000 --naechte 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy import rules                                    # noqa: E402
from strategy.sg_api import SGClient                          # noqa: E402

CLS_DISCOUNT_CLASSIC = 21


def finde_parkpapiere(asset_id: int = 389, mindest_puffer: float = 0.05,
                      anzahl: int = 8) -> list[dict]:
    """Sucht Discount-Zertifikate mit sicherem Abstand zum Cap.

    `Distance2CapLevelPercent` ist negativ, wenn der Basiswert ueber dem Cap
    liegt -- genau das ist der gewuenschte Zustand.
    """
    c = SGClient()
    j = c.search(CLS_DISCOUNT_CLASSIC, asset_id, page_size=300, page_number=1)
    treffer = []
    for r in j.get("Products", []):
        if not ((r.get("Bid") or 0) > 0 and (r.get("Offer") or 0) > 0):
            continue
        p = c.properties_dict(r["Id"])
        puffer = p.get("Distance2CapLevelPercent")
        rendite = p.get("MaxPerformanceStrikeCurrencyPercentPA")
        if puffer is None or rendite is None or puffer > -mindest_puffer * 100:
            continue
        treffer.append({"wkn": r["Code"], "brief": r["Offer"],
                        "spread": (r["Offer"] - r["Bid"]) / r["Offer"],
                        "puffer_%": -puffer, "rendite_pa_%": rendite,
                        "faellig": str(p.get("MaturityDate"))[:10]})
        if len(treffer) >= anzahl:
            break
    return sorted(treffer, key=lambda x: -x["rendite_pa_%"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cash", type=float, default=200_000.0,
                    help="brachliegendes Kapital ueber beide Depots")
    ap.add_argument("--depots", type=int, default=2)
    ap.add_argument("--naechte", type=int, default=1)
    ap.add_argument("--offline", action="store_true",
                    help="ohne Abruf rechnen, mit 2,71 %% und 0,004 %% Spread")
    args = ap.parse_args()

    if args.offline:
        zins, spread, papiere = 0.0271, 0.00004, []
    else:
        papiere = finde_parkpapiere()
        if not papiere:
            print("Keine geeigneten Discount-Zertifikate gefunden.")
            return
        zins = papiere[0]["rendite_pa_%"] / 100
        spread = papiere[0]["spread"]
        print("Geeignete Parkpapiere (Cap sicher unter dem Basiswertkurs):\n")
        print(f"  {'WKN':<9}{'Brief':>9}{'Puffer':>9}{'Rendite p.a.':>14}"
              f"{'Spread':>9}{'faellig':>13}")
        for p in papiere[:5]:
            print(f"  {p['wkn']:<9}{p['brief']:>9.2f}{p['puffer_%']:>8.2f}%"
                  f"{p['rendite_pa_%']:>13.2f}%{p['spread']*100:>8.3f}%"
                  f"{p['faellig']:>13}")
        print()

    je_depot = args.cash / args.depots
    n_papiere = max(1, int(-(-je_depot // rules.max_position_eur(je_depot))))
    orders = n_papiere * 2 * args.depots
    gebuehren = orders * rules.FEE_DERIVATIVE
    spreadkosten = args.cash * spread
    kosten = gebuehren + spreadkosten
    pro_nacht = args.cash * zins / 365

    print(f"  Kapital                       {args.cash:>11,.0f} EUR")
    print(f"  Zins                          {zins*100:>11.2f} % p.a.")
    print(f"  Ertrag je Nacht               {pro_nacht:>11.2f} EUR")
    print(f"  20-%-Regel: {n_papiere} Papiere je Depot -> {orders} Orders")
    print(f"  Gebuehren                     {gebuehren:>11.2f} EUR")
    print(f"  Spread                        {spreadkosten:>11.2f} EUR")
    print(f"  Break-even nach               {kosten/pro_nacht:>11.1f} Naechten")
    netto = pro_nacht * args.naechte - kosten
    print(f"\n  Bei {args.naechte} Nacht/Naechten: {netto:>+9.2f} EUR "
          f"-> {'lohnt sich' if netto > 0 else 'lohnt sich NICHT'}")
    if netto <= 0:
        print(f"  Erst ab {int(kosten/pro_nacht)+1} Naechten im Plus.")
    print("\n  Nicht risikofrei: Faellt der Basiswert unter den Cap, wird aus")
    print("  der Parkposition eine ungewollte Wette auf den Basiswert.")


if __name__ == "__main__":
    main()
