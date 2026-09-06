#!/usr/bin/env python3
"""Durchsucht das gesamte SG-Basiswertuniversum nach der besten Konvexitaet.

Entscheidend ist nicht der Hebel allein, sondern **Hebel x Tagesvola des
Basiswerts** -- die Bewegung, die die 20.000 EUR an einem Tag machen koennen.
Ein DAX-Turbo mit Hebel 75 auf einem Basiswert mit 0,56 % Tagesvola bewegt
sich weniger als ein NVIDIA-Turbo mit Hebel 33 auf 2,87 %.

Das Ticker-Mapping von SGs RIC auf Yahoo ist unzuverlaessig (RHMG.DE gegen
RHM.DE, .SPX gegen ^GSPC). Deshalb wird jedes Mapping gegen den aus den
Produktdaten abgeleiteten Basiswertkurs geprueft: Weicht der Yahoo-Kurs um
mehr als die Toleranz ab, gilt das Mapping als falsch und der Basiswert
faellt raus, statt mit fremder Volatilitaet bewertet zu werden.

    python3 scripts/scan_universe.py --region eu --max-assets 80
"""

from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy import data                                        # noqa: E402
from strategy.sg_api import (CLS_BEST_TURBO_OPEN_END, SGClient,   # noqa: E402
                             fx_to_eur, infer_spot, to_turbos, validate_pricing)

EU_SUFFIX = {"DE": "DE", "PA": "PA", "AS": "AS", "MI": "MI", "L": "L", "S": "SW",
             "ST": "ST", "CO": "CO", "OL": "OL", "HE": "HE", "VI": "VI",
             "MC": "MC", "BR": "BR", "LS": "LS", "F": "F"}
US_SUFFIX = {"OQ", "N", "O", "A", "K", "P"}

# Indizes mappen fast durchgaengig als "^" + RIC-Wurzel; die Ausnahmen hier.
INDEX_MAP = {".SPX": "^GSPC", ".FTMIB": "FTSEMIB.MI", ".STOXX50E": "^STOXX50E",
             ".HSCE": "^HSCE", ".HSTECH": "^HSTECH"}


def yahoo_kandidaten(ric: str) -> list[str]:
    """Moegliche Yahoo-Symbole zu einem RIC, beste Vermutung zuerst."""
    if not ric:
        return []
    if ric.startswith("."):
        return [INDEX_MAP.get(ric, "^" + ric[1:])]
    if "." not in ric:
        return [ric]
    wurzel, suf = ric.rsplit(".", 1)
    if suf in US_SUFFIX:
        return [wurzel]
    y = EU_SUFFIX.get(suf)
    if not y:
        return []
    kand = [f"{wurzel}.{y}"]
    # Deutsche RICs haengen oft ein G oder n an (SAPG.DE -> SAP.DE)
    for endung in ("G", "n", "d", "e"):
        if wurzel.endswith(endung) and len(wurzel) > 2:
            kand.append(f"{wurzel[:-1]}.{y}")
    return kand


def loese_ticker(ric: str, spot_eur: float, fx: float,
                 toleranz: float = 0.05) -> tuple[str | None, float | None]:
    """Sucht das Yahoo-Symbol und prueft es gegen den bekannten Kurs."""
    for sym in yahoo_kandidaten(ric):
        df = data.history(sym, rng="6mo", interval="1d", cache_minutes=1440)
        if df.empty or len(df) < 25:
            continue
        kurs = float(df["close"].iloc[-1])
        waehrung = df.attrs.get("currency")
        kurs_eur = kurs / fx if waehrung == "USD" else kurs
        if abs(kurs_eur - spot_eur) / max(spot_eur, 1e-9) <= toleranz:
            vola = float(df["close"].pct_change().tail(20).std())
            return sym, vola
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", choices=["eu", "us", "index", "alle"], default="eu")
    ap.add_argument("--max-assets", type=int, default=80)
    ap.add_argument("--min-produkte", type=int, default=60,
                    help="Basiswerte mit weniger Turbos ueberspringen (Liquiditaet)")
    ap.add_argument("--max-spread", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="data/universum_scan.csv")
    args = ap.parse_args()

    c = SGClient()
    assets = c.assets()

    def region_von(a) -> str:
        ric = str(a.get("Ric") or "")
        if ric.startswith("."):
            return "index"
        suf = ric.rsplit(".", 1)[-1] if "." in ric else ""
        if suf in US_SUFFIX:
            return "us"
        return "eu" if suf in EU_SUFFIX else "sonst"

    kand = [a for a in assets if args.region == "alle" or region_von(a) == args.region]
    print(f"{len(kand)} Basiswerte in Region '{args.region}', pruefe bis zu {args.max_assets}")

    clients = [SGClient(pause=0.4 * args.workers) for _ in range(args.workers)]

    def pruefe(i_a):
        i, a = i_a
        cl = clients[i % args.workers]
        try:
            j = cl.search(CLS_BEST_TURBO_OPEN_END, a["Id"], page_size=250, page_number=1)
        except Exception:
            return None
        recs = j.get("Products", [])
        if j.get("TotalCount", 0) < args.min_produkte or not recs:
            return None
        fx = fx_to_eur(recs)
        spot = infer_spot(recs, fx)
        if not spot or spot <= 0:
            return None
        ts = to_turbos(recs, spot=spot, verbose=False)
        ts, _ = validate_pricing(ts, spot, verbose=False)
        ts = [t for t in ts if t.product_type == "turbo" and t.spread_pct <= args.max_spread]
        if len(ts) < 10:
            return None
        eng = min(ts, key=lambda t: t.distance_to_ko_pct)
        sym, vola = loese_ticker(str(a.get("Ric") or ""), spot, fx)
        if not sym or not vola:
            return None
        return {"Basiswert": a["Name"], "Yahoo": sym, "Produkte": j["TotalCount"],
                "Kurs EUR": spot, "Tagesvola %": vola * 100,
                "engster KO %": eng.distance_to_ko_pct * 100,
                "in Sigma": eng.distance_to_ko_pct / vola,
                "Hebel": eng.omega, "Spread %": eng.spread_pct * 100,
                "Sleeve-Tagesvola %": eng.omega * vola * 100, "WKN": eng.wkn}

    zeilen = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, res in enumerate(pool.map(pruefe, enumerate(kand[:args.max_assets])), 1):
            if res:
                zeilen.append(res)
            if n % 20 == 0:
                print(f"  {n}/{min(len(kand), args.max_assets)} geprueft, "
                      f"{len(zeilen)} verwertbar")

    if not zeilen:
        print("Keine verwertbaren Basiswerte.")
        return
    df = pd.DataFrame(zeilen).sort_values("Sleeve-Tagesvola %", ascending=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    print(f"\n{len(df)} Basiswerte bewertet -> {args.out}\n")
    print(df.head(25).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":
    main()
