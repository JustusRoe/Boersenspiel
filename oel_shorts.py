"""Alle Oel-Shorts, ohne die Vorfilter, die die scharfen Produkte wegwerfen.

Frueheren Scans habe ich zwei Bedingungen mitgegeben, die genau die
hoechstgehebelten Produkte aussortieren:
  * Spread <= 2 %  -- filtert die barrierennahen Scheine weg, bei denen der
    Emittent breiter stellt,
  * "97 % des Hebeltopfs muessen in EIN Produkt passen" -- filtert alles
    unter etwa 0,94 EUR weg, weil dort die 20.000-Stueck-Grenze zuerst
    bindet.
Zusammen haben sie die Antwort "mehr als Hebel 30 gibt es nicht" erzeugt,
die schlicht falsch war.

Drei Dinge, die dieser Scan anders macht:

1. Kurs je Terminkontrakt. SG fuehrt siebzehn Oel-Kontrakte unter zwei
   Namen; die Kurve ist stark in Backwardation (Brent Nov 26 bei 99,
   Dez 27 bei 80). Wer alle Produkte gegen einen einzigen Kurs rechnet,
   haelt Classic Turbos auf ferne Kontrakte fuer bereits ausgeknockt.

2. Kurse aus den Detaildaten, nicht aus der Trefferliste. In der
   Trefferliste fehlt der Geldkurs bei vielen Produkten, was sie faelschlich
   als unverkaeuflich aussehen laesst -- und umgekehrt stehen dort Kurse fuer
   Scheine, die laengst ausgeknockt sind.

3. Offen: die Unlimited Turbos (Mini) notieren nach diesem Modell unter
   ihrem inneren Wert (negatives Aufgeld in der Tabelle). Entweder liegt die
   Wertbasis anders als im Feld `Strike`, oder sie sind nicht Quanto. Fuer die
   Auswahl spielt es keine Rolle -- sie haben mit Hebel 22 ohnehin ein
   Fuenftel des Hebels der BEST Turbos --, aber die Zahlen in ihren Zeilen
   sind nicht belastbar.

4. Kein Rueckgriff auf `CurrentLeverage`. Das Feld ist bei den Oelscheinen
   veraltet: fuer FC8LW4 steht dort 30,05, was einem Brent-Kurs von 77,8
   entspraeche -- Brent steht bei 99. Der Hebel folgt ohnehin aus Barriere
   und Preis.
"""
from __future__ import annotations
import argparse, json, math, random, statistics, sys

from strategy import sg_api as sg
from strategy import data
from strategy.rules import leverage_budget

KLASSEN = (sg.CLS_BEST_TURBO_OPEN_END, sg.CLS_UNLIMITED_TURBO_MINI,
           sg.CLS_CLASSIC_TURBO, sg.CLS_STANDARD_OS)
BASISWERTE = {-4: ("Brent", "BZ=F"), -5: ("WTI", "CL=F")}

# SG fuehrt beide Rohoelsorten in je acht Terminkontrakten unter einem
# Basiswertnamen. Die Kurve steht steil in Backwardation -- Brent Nov 26
# notiert bei 99, Dez 27 bei 80. Wer alle Produkte gegen den Frontkurs
# rechnet, haelt einen Classic Turbo auf Dez 27 mit Barriere 100 fuer einen
# Schein mit 1 % Abstand, waehrend er in Wahrheit 25 % aus dem Geld liegt.
# Nur die Frontkontrakte tragen die Tagesvolatilitaet, auf die es hier
# ankommt, also bleiben nur sie uebrig.
def kontraktkurs(saetze) -> float | None:
    """Leitet den Kurs eines Terminkontrakts aus seinen eigenen Produkten ab.

    Die Trefferliste enthaelt den Basiswertkurs nicht, wohl aber Barriere und
    Preis jedes Scheins. Fuer einen Short gilt Preis = Barriere - Kurs (Quanto,
    Ratio 1), also Kurs = Barriere - Preis. Der Median ueber alle Shorts eines
    Kontrakts ist robust gegen einzelne veraltete Kurse.
    """
    werte = []
    for r in saetze:
        if str(r.get("PutOrCall") or "").lower() not in ("put", "short"):
            continue
        barriere = zahl(r.get("BarrierTurboCertificate")) or zahl(r.get("Strike"))
        geld, brief = zahl(r.get("Bid")), zahl(r.get("Offer"))
        mitte = (geld + brief) / 2 if geld > 0 and brief > 0 else (geld or brief)
        ratio = zahl(r.get("Ratio")) or 1.0
        if barriere > 0 and mitte > 0:
            werte.append(barriere - mitte * ratio)
    return statistics.median(werte) if len(werte) >= 5 else None

# Preismodell der Brent-Short-Leiter, regressiert ueber 43 eng gestellte
# Produkte (R2 = 0,96): der konstante Term ist die Gap-Praemie des
# Emittenten. Sie faellt JE SCHEIN an -- und weil die Stueckzahl bei 20.000
# gedeckelt ist, frisst sie bei billigen Scheinen den halben Einsatz.
STEIGUNG, GAP_PRAEMIE = 0.93, 0.33


def zahl(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if f != f else f


def tagesmuster(symbol: str, rng: str = "6mo"):
    """Ganze historische Tage (Open/Hoch/Tief/Schluss je Vortagesschluss).

    Der Knock-out haengt am Tageshoch, nicht am Schlusskurs. Eine
    Normalverteilung unterschaetzt Eroeffnungsluecken systematisch, deshalb
    werden echte Tage mit Zuruecklegen gezogen.
    """
    h = data.history(symbol, rng=rng).dropna(subset=["open", "high", "low", "close"])
    o, hi, lo, cl = (h[k].tolist() for k in ("open", "high", "low", "close"))
    return [(o[i]/cl[i-1]-1, hi[i]/cl[i-1]-1, lo[i]/cl[i-1]-1, cl[i]/cl[i-1]-1)
            for i in range(1, len(cl)) if cl[i-1] > 0]


def tagessimulation(muster, spot, barriere, brief, topf, depot,
                    basispreis=None, n=300_000, seed=13):
    """basispreis: Wertbasis des Scheins. Bei BEST Turbos gleich der Barriere,
    bei Unlimited Turbos (Mini) liegt sie darueber -- wer beide gleich setzt,
    haelt einen Mini fuer masslos ueberteuert (im Test 14.000 EUR "Aufgeld"),
    obwohl er nur schlicht weniger Hebel hat.
    """
    basis = basispreis if basispreis and basispreis > barriere else barriere
    stueck = min(int(topf // brief), 20_000)
    einsatz = stueck * brief
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        _, hoch, _, schluss = muster[rnd.randrange(len(muster))]
        if spot * (1 + hoch) >= barriere:          # Barriere intraday gerissen
            # Beim Mini wird zum Restwert glattgestellt, nicht auf null.
            wert = stueck * max(basis - spot*(1+hoch), 0.0) * 0.9 if basis > barriere else 0.0
        else:
            wert = stueck * max(STEIGUNG * (basis - spot*(1+schluss)) + GAP_PRAEMIE, 0.0)
        out.append((wert - einsatz) / depot)
    out.sort()
    anteil = lambda x: sum(1 for r in out if r >= x) / n
    return dict(stueck=stueck, einsatz=einsatz,
                aufgeld=stueck * (brief - max(basis - spot, 0.0)),
                p_ko=sum(1 for r in out if r <= -einsatz/depot + 1e-9) / n,
                median=out[n//2], p90=out[int(.90*n)], p99=out[int(.99*n)],
                ge25=anteil(.25), ge50=anteil(.50), ge100=anteil(1.0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depot", type=float, default=93_500.0)
    ap.add_argument("--pruefen", type=int, default=70,
                    help="wie viele barrierennahe Shorts live nachgefragt werden")
    ap.add_argument("--csv", default="data/oel_shorts.csv")
    args = ap.parse_args()

    topf = leverage_budget(args.depot)
    print(f"Depot {args.depot:,.0f} EUR  ->  Hebeltopf {topf:,.0f} EUR "
          f"({topf/args.depot*100:.0f} % des Depots)")
    print(f"Optimaler Preispunkt = Topf / 20.000 Stueck = {topf/20_000:.3f} EUR\n")

    client = sg.SGClient(pause=0.3)
    kurse, muster = {}, {}
    for aid, (name, symbol) in BASISWERTE.items():
        kurse[name] = data.last_price(symbol)
        muster[name] = tagesmuster(symbol)
        print(f"{name}: Kurs {kurse[name]:.2f}, {len(muster[name])} Tage Muster")

    treffer = {}
    for aid, (name, _) in BASISWERTE.items():
        for cls in KLASSEN:
            for r in client.iter_products(cls, aid, page_size=1000, verbose=False):
                r["_basis"] = name
                treffer[r.get("Code")] = r
    print(f"\n{len(treffer):,} Oel-Produkte geladen")

    # Auf den Frontkontrakt eindampfen (siehe Kommentar bei kontraktkurs).
    kontrakte: dict[str, list] = {}
    for r in treffer.values():
        kontrakte.setdefault(r.get("AssetName"), []).append(r)
    front = {}
    for basis in kurse:
        eigene = {n: v for n, v in kontrakte.items() if any(r["_basis"] == basis for r in v)}
        bewertet = {n: kontraktkurs(v) for n, v in eigene.items()}
        bewertet = {n: k for n, k in bewertet.items() if k}
        if not bewertet:
            continue
        front[basis] = min(bewertet, key=lambda n: abs(bewertet[n] - kurse[basis]))
        print(f"  {basis}: {len(eigene)} Kontrakte, Front = '{front[basis]}' "
              f"(impliziert {bewertet[front[basis]]:.2f} gegen Yahoo {kurse[basis]:.2f}); "
              f"verworfen: {', '.join(sorted(n.split('Future')[-1].strip() for n in eigene if n != front[basis]))}")
    treffer = {c: r for c, r in treffer.items() if r.get("AssetName") == front.get(r["_basis"])}
    print(f"{len(treffer):,} Produkte auf den Frontkontrakten\n")

    # Kurs je Terminkontrakt: die Frontkontrakte tragen die Volatilitaet,
    # die fernen Kontrakte notieren zwanzig Dollar tiefer.
    kandidaten = []
    for r in treffer.values():
        if str(r.get("PutOrCall") or "").lower() not in ("put", "short"):
            continue
        basis = r["_basis"]
        spot = kurse[basis]
        barriere = zahl(r.get("BarrierTurboCertificate")) or zahl(r.get("Strike"))
        if not (spot * 1.001 < barriere < spot * 1.08):
            continue                       # tot, oder so weit weg, dass kein Hebel bleibt
        kandidaten.append((barriere - spot, r, basis, spot, barriere))
    kandidaten.sort(key=lambda x: x[0])
    print(f"{len(kandidaten):,} lebende Shorts nahe der Barriere; "
          f"frage die {min(args.pruefen, len(kandidaten))} schaerfsten live ab\n")

    zeilen = []
    for _, r, basis, spot, barriere in kandidaten[:args.pruefen]:
        try:
            p = client.properties_dict(r["Id"])
        except sg.SGApiError:
            continue
        geld, brief = zahl(p.get("Bid")), zahl(p.get("Offer"))
        barriere = zahl(p.get("BarrierTurboCertificate")) or barriere
        if geld <= 0.01 or brief <= 0.01:
            zeilen.append(dict(wkn=r["Code"], basis=basis, barriere=barriere,
                               status="ausgeknockt"))
            continue
        s = tagessimulation(muster[basis], spot, barriere, brief, topf, args.depot,
                            basispreis=zahl(p.get("Strike")))
        zeilen.append(dict(wkn=r["Code"], basis=basis, barriere=barriere, status="handelbar",
                           abstand=(barriere-spot)/spot, geld=geld, brief=brief,
                           spread=(brief-geld)/brief, hebel=spot/brief, **s))

    tot = [z for z in zeilen if z["status"] == "ausgeknockt"]
    lebt = [z for z in zeilen if z["status"] == "handelbar"]
    print(f"{len(lebt)} handelbar, {len(tot)} inzwischen ausgeknockt "
          f"({', '.join(z['wkn'] for z in tot[:10])}{' ...' if len(tot) > 10 else ''})\n")

    kopf = (f"{'WKN':8}{'Basis':7}{'Barriere':>9}{'Abst%':>7}{'Geld':>6}{'Brief':>6}"
            f"{'Spr%':>6}{'Hebel':>6}{'Stueck':>8}{'Einsatz':>9}{'Aufgeld':>9}"
            f"{'P(KO)':>7}{'Median':>8}{'P99':>8}{'>=50%':>7}{'>=100%':>8}")
    print(kopf)
    for z in sorted(lebt, key=lambda z: -z["ge100"])[:20]:
        print(f"{z['wkn']:8}{z['basis']:7}{z['barriere']:9.2f}{z['abstand']*100:6.2f}%"
              f"{z['geld']:6.2f}{z['brief']:6.2f}{z['spread']*100:6.1f}{z['hebel']:6.0f}"
              f"{z['stueck']:8,}{z['einsatz']:9,.0f}{z['aufgeld']:9,.0f}{z['p_ko']*100:6.1f}%"
              f"{z['median']*100:7.1f}%{z['p99']*100:7.1f}%{z['ge50']*100:6.1f}%{z['ge100']*100:7.1f}%")

    if args.csv:
        import csv as _csv
        felder = sorted({k for z in zeilen for k in z})
        with open(args.csv, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=felder)
            w.writeheader()
            w.writerows(zeilen)
        print(f"\n-> {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
