#!/usr/bin/env python3
"""Laedt Produktdaten direkt aus der SG-API statt aus dem Excel-Export.

Der Excel-Export ist auf 5.000 Produkte gedeckelt und liefert weder
Knock-out-Schwelle noch Bezugsverhaeltnis noch Long/Short -- damit ist kein
Turbo bewertbar. Die API liefert genau diese Felder mit.

Beispiele:
    python3 scripts/sg_scrape.py --assets "DAX 40" --typen turbo
    python3 scripts/sg_scrape.py --assets "DAX 40,Nasdaq-100,S&P 500" \\
            --typen turbo,factor --max 4000
    python3 scripts/sg_scrape.py --suche NVIDIA          # Asset-ID herausfinden
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy.sg_api import (ASSETS, CLS_BEST_TURBO_OPEN_END,  # noqa: E402
                             CLS_FAKTOR_OS_LEAF, CLS_STANDARD_OS,
                             CLS_UNLIMITED_TURBO_MINI, SGClient,
                             enrich_factors, infer_spot)

TYP_GRUPPEN = {
    "turbo":  [CLS_BEST_TURBO_OPEN_END, CLS_UNLIMITED_TURBO_MINI],
    "factor": [CLS_FAKTOR_OS_LEAF],
    "warrant": [CLS_STANDARD_OS],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="DAX 40",
                    help="Kommaliste von Basiswerten, z.B. 'DAX 40,S&P 500'")
    ap.add_argument("--typen", default="turbo,factor",
                    help=f"Kommaliste aus {sorted(TYP_GRUPPEN)}")
    ap.add_argument("--max", type=int, default=3000,
                    help="max. Produkte je Basiswert und Klassifikation")
    ap.add_argument("--out", default="data/sg/sg_products.csv")
    ap.add_argument("--suche", help="Nur Basiswerte suchen und Ids ausgeben")
    ap.add_argument("--pause", type=float, default=0.4,
                    help="Sekunden zwischen Requests (Selbstdrosselung)")
    args = ap.parse_args()

    c = SGClient(pause=args.pause)

    if args.suche:
        treffer = c.find_asset(args.suche)
        if not treffer:
            print(f"Kein Basiswert enthaelt '{args.suche}'.")
            return
        print(f"{len(treffer)} Treffer:")
        for a in treffer[:40]:
            print(f"  Id={a['Id']:>6}  {a['Name']}   (RIC {a.get('Ric')})")
        return

    alle_assets = c.assets()
    nach_name = {a["Name"]: a["Id"] for a in alle_assets}

    rows: list[dict] = []
    for asset_name in [a.strip() for a in args.assets.split(",") if a.strip()]:
        aid = ASSETS.get(asset_name) or nach_name.get(asset_name)
        if aid is None:
            treffer = [a for a in alle_assets if asset_name.upper() in a["Name"].upper()]
            if not treffer:
                print(f"[!] Basiswert '{asset_name}' nicht gefunden -- "
                      f"mit --suche danach suchen. Uebersprungen.")
                continue
            aid, asset_name = treffer[0]["Id"], treffer[0]["Name"]
            print(f"[i] '{asset_name}' (Id {aid}) verwendet")

        for typ in [t.strip() for t in args.typen.split(",") if t.strip()]:
            if typ not in TYP_GRUPPEN:
                print(f"[!] Typ '{typ}' unbekannt, uebersprungen")
                continue
            for cls in TYP_GRUPPEN[typ]:
                try:
                    recs = list(c.iter_products(cls, aid, page_size=1000,
                                                max_products=args.max))
                except Exception as exc:
                    print(f"[!] {asset_name}/{cls}: {exc}")
                    continue
                if typ == "factor":
                    # Hebel und Richtung stehen bei Faktor-Produkten nicht in
                    # der Trefferliste, sondern nur in den Detaildaten.
                    recs = enrich_factors(c, recs)
                for r in recs:
                    r["_asset_name"] = asset_name
                    r["_typ"] = typ
                rows.extend(recs)

    if not rows:
        print("Nichts geladen.")
        return

    df = pd.DataFrame(rows).drop_duplicates(subset=["Code"], keep="first")
    # Basiswertkurs je Asset ableiten (die Trefferliste enthaelt ihn nicht)
    for asset_name, grp in df.groupby("_asset_name"):
        spot = infer_spot(grp.to_dict("records"))
        df.loc[df["_asset_name"] == asset_name, "_spot"] = spot
        print(f"  {asset_name}: {len(grp):,} Produkte, abgeleiteter Kurs {spot:,.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n{len(df):,} Produkte -> {out}")
    handelbar = int(((df.get("Offer", 0)).fillna(0) > 0).sum())
    print(f"davon mit Briefkurs (kaufbar): {handelbar:,}")


if __name__ == "__main__":
    main()
