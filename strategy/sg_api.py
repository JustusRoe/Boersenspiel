"""Client fuer die JSON-API hinter sg-zertifikate.de.

Die Produktsuche der Societe Generale ist eine Angular-SPA; die Seite selbst
enthaelt keine Produktdaten. Der Browser spricht mit einer REST-API unter
`/emcwebapi/api/`, deren Basis-URL im Frontend-Bundle als
`${baseUrls.emcWebApi}/api/${endpoint}` zusammengesetzt wird.

Warum dieser Weg dem Excel-Export vorzuziehen ist: Der Export ist auf 5.000
Produkte gedeckelt (bei rund 300.000 im Universum) und liefert weder
Knock-out-Schwelle noch Bezugsverhaeltnis noch die Long/Short-Richtung --
damit ist kein Turbo bewertbar. `ProductSearch/Search` liefert genau diese
Felder mit: PutOrCall, Ratio, Strike, BarrierTurboCertificate,
CurrentLeverage, Bid und Offer.

Der Client ruft ausschliesslich oeffentliche, unauthentifizierte Endpunkte
auf, drosselt sich selbst und laedt nur die angeforderten Basiswerte.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests

BASE = "https://www.sg-zertifikate.de/emcwebapi/api/"

# Produktklassifikationen (aus ProductSearch/ProductClassifications)
CLS_OPTIONSSCHEINE = 1
CLS_STANDARD_OS = 2
CLS_INLINE_OS = 8
CLS_TURBO = 42
CLS_CLASSIC_TURBO = 43
CLS_UNLIMITED_TURBO_MINI = 45
CLS_BEST_TURBO_OPEN_END = 47
CLS_SMART_TURBO = 49
CLS_STOP_LOSS_TURBO = 50939
CLS_FAKTOR_OS = 228            # Oberkategorie
CLS_FAKTOR_OS_LEAF = 44100     # tatsaechliche Klassifikation der Produkte

# Fuer das Boersenspiel relevante Klassifikationen mit Modelltyp
SPIELRELEVANT = {
    CLS_BEST_TURBO_OPEN_END: "turbo",
    CLS_UNLIMITED_TURBO_MINI: "turbo",
    CLS_CLASSIC_TURBO: "turbo",
    CLS_STOP_LOSS_TURBO: "turbo",
    CLS_SMART_TURBO: "turbo",
    CLS_FAKTOR_OS: "factor",
    CLS_FAKTOR_OS_LEAF: "factor",
    CLS_STANDARD_OS: "warrant",
}

# Haeufig gebrauchte Basiswerte (ProductSearch/Assets)
ASSETS = {
    "DAX 40": 389, "MDAX": 141, "TecDAX": 233,
    "Nasdaq-100": 315, "S&P 500": 693,
}


class SGApiError(RuntimeError):
    pass


@dataclass
class SGClient:
    """Duenner Client mit Wiederholung und Selbstdrosselung."""
    pause: float = 0.4          # Sekunden zwischen Requests
    timeout: int = 60
    tries: int = 4

    def __post_init__(self):
        self.s = requests.Session()
        self.s.headers.update({
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"),
            "Accept": "application/json",
            "Accept-Language": "de-DE,de;q=0.9",
            "Referer": "https://www.sg-zertifikate.de/",
        })
        self._last = 0.0

    def _get(self, endpoint: str, params: dict | None = None):
        last_err = None
        for attempt in range(self.tries):
            delta = time.time() - self._last
            if delta < self.pause:
                time.sleep(self.pause - delta)
            try:
                r = self.s.get(BASE + endpoint, params=params, timeout=self.timeout)
                self._last = time.time()
                if r.status_code == 200:
                    return r.json()
                last_err = f"HTTP {r.status_code}"
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(1.5 * (attempt + 1))
        raise SGApiError(f"{endpoint} fehlgeschlagen: {last_err}")

    # ------------------------------------------------------------ Metadaten
    def classifications(self) -> list[dict]:
        return self._get("ProductSearch/ProductClassifications")

    def assets(self, classification_id: int = CLS_BEST_TURBO_OPEN_END) -> list[dict]:
        return self._get("ProductSearch/Assets", {"productClassificationId": classification_id})

    def find_asset(self, name_fragment: str,
                   classification_id: int = CLS_BEST_TURBO_OPEN_END) -> list[dict]:
        frag = name_fragment.upper()
        return [a for a in self.assets(classification_id)
                if frag in str(a.get("Name", "")).upper()]

    # -------------------------------------------------------------- Produkte
    def search(self, classification_id: int, asset_id: int | None = None,
               page_size: int = 1000, page_number: int = 1, **extra) -> dict:
        # Achtung: Die API ist bei den Parameternamen eigenwillig. Nur
        # `pageNum` blaettert tatsaechlich -- `PageNumber`, `pageNumber`,
        # `pageIndex`, `skip` und `offset` werden stillschweigend ignoriert
        # und liefern immer wieder die erste Seite. Das faellt nicht auf,
        # weil weiterhin HTTP 200 mit vollen Trefferlisten zurueckkommt.
        params = {"productClassificationId": classification_id,
                  "pageSize": page_size, "pageNum": page_number}
        if asset_id is not None:
            params["assetId"] = asset_id
        params.update(extra)
        return self._get("ProductSearch/Search", params)

    def iter_products(self, classification_id: int, asset_id: int | None = None,
                      page_size: int = 1000, max_products: int | None = None,
                      verbose: bool = True):
        """Blaettert durch die Trefferliste. Laedt nur, was angefordert wurde."""
        first = self.search(classification_id, asset_id, page_size, 1)
        total = first.get("TotalCount", 0)
        limit = min(total, max_products) if max_products else total
        if verbose:
            print(f"  Klassifikation {classification_id}"
                  f"{f', Asset {asset_id}' if asset_id else ''}: "
                  f"{total:,} Treffer, lade {limit:,}")
        yielded = 0
        gesehen: set[str] = set()
        for prod in first.get("Products", []):
            if yielded >= limit:
                return
            gesehen.add(prod.get("Code"))
            yield prod
            yielded += 1
        page = 2
        while yielded < limit:
            batch = self.search(classification_id, asset_id, page_size, page)
            prods = batch.get("Products", [])
            if not prods:
                return
            neu = 0
            for prod in prods:
                if yielded >= limit:
                    return
                code = prod.get("Code")
                if code in gesehen:            # Schutz vor stiller Doppelung
                    continue
                gesehen.add(code)
                neu += 1
                yield prod
                yielded += 1
            if neu == 0:                       # Seite brachte nichts Neues
                return
            page += 1

    def all_properties(self, product_id: int) -> list[dict]:
        """Vollstaendige Kennzahlen eines Produkts (Strike, Barriere, Delta ...)."""
        return self._get(f"Products/AllProperties/{product_id}")

    def properties_dict(self, product_id: int) -> dict:
        return {p["Name"]: p.get("Value") for p in self.all_properties(product_id)
                if isinstance(p, dict) and "Name" in p}

    def products_by_codes(self, codes: list[str] | str) -> list[dict]:
        if isinstance(codes, list):
            codes = ",".join(codes)
        return self._get("Products/GetProductsByCodes", {"codes": codes})

    def price_history(self, product_id: int) -> list:
        return self._get(f"Prices/{product_id}")


# ------------------------------------------------------------ Konvertierung
# Yahoo-Symbole fuer "Einheiten der Waehrung je EUR"
FX_SYMBOLE = {"USD": "EURUSD=X", "JPY": "EURJPY=X", "GBP": "EURGBP=X",
              "CHF": "EURCHF=X", "SEK": "EURSEK=X", "NOK": "EURNOK=X",
              "DKK": "EURDKK=X", "CAD": "EURCAD=X", "AUD": "EURAUD=X",
              "HKD": "EURHKD=X", "PLN": "EURPLN=X"}
# Waehrungen, die keine sind: Indexpunkte und Prozentnotierungen
KEINE_WAEHRUNG = {"", "EUR", "Pkt", "PKT", "%", "Punkte", "Index"}


def fx_to_eur(records: list[dict], fx_rate: float | None = None) -> float:
    """Umrechnungsfaktor: Einheiten der Basiswertwaehrung je EUR.

    Der Basispreis steht in der Waehrung des Basiswerts, der Schein notiert
    in EUR. Ohne Umrechnung ist der theoretische Preis um den Wechselkurs
    daneben. Das faellt nicht als Fehler auf, sondern verschiebt die
    Preisprobe -- bei USD um Faktor 1,16, bei JPY um Faktor 180.

    Der Rueckgabewert ist immer "wie viele Einheiten der Basiswertwaehrung
    entsprechen einem Euro", also 1,16 fuer USD und rund 180 fuer JPY.
    Indexpunkte und Prozentnotierungen liefern 1,0.
    """
    waehrungen = {str(r.get("AssetCurrency") or r.get("AssetCurrencyRaw") or "").strip()
                  for r in records}
    fremd = {w for w in waehrungen if w not in KEINE_WAEHRUNG}
    if not fremd:
        return 1.0
    if len(fremd) > 1:
        print(f"  [fx] uneinheitliche Basiswertwaehrungen {sorted(fremd)} -- "
              f"keine Umrechnung, Ergebnisse pruefen")
        return 1.0
    ccy = fremd.pop()
    if fx_rate is not None:
        return float(fx_rate)
    sym = FX_SYMBOLE.get(ccy)
    if not sym:
        print(f"  [fx] kein Wechselkurs fuer '{ccy}' hinterlegt -- keine Umrechnung")
        return 1.0
    from . import data
    kurs = data.last_price(sym)
    if not kurs:
        print(f"  [fx] Wechselkurs {sym} nicht abrufbar -- keine Umrechnung")
        return 1.0
    return float(kurs)


def infer_spot(records: list[dict], fx: float = 1.0) -> float | None:
    """Leitet den Basiswertkurs aus Strike und Produktpreis ab.

    Die Trefferliste enthaelt den Kurs des Basiswerts nicht, wohl aber alles,
    woraus er exakt folgt. Achtung bei der Konvention: SG gibt `Ratio` als
    Divisor an (Bezugsverhaeltnis "100:1" heisst, 100 Scheine beziehen sich
    auf eine Einheit des Basiswerts), also gilt

        Preis = |Spot - Strike| / Ratio_SG   =>   Spot = Strike +- Preis * Ratio_SG

    Der Umweg ueber den Hebel (Spot = Hebel * Preis * Ratio) waere ebenfalls
    moeglich, ist aber deutlich ungenauer: der Hebel wird vom Emittenten zu
    einem eigenen Zeitpunkt gerechnet und lag im Test 1,9 % daneben, die
    Strike-Formel dagegen 0,07 %.
    """
    import statistics
    kandidaten = []
    for r in records:
        strike = r.get("Strike")
        ratio = r.get("Ratio") or 1.0
        bid, offer = r.get("Bid") or 0.0, r.get("Offer") or 0.0
        px = (bid + offer) / 2 if bid > 0 and offer > 0 else (bid or offer)
        if not strike or px <= 0 or ratio <= 0:
            continue
        d = -1 if str(r.get("PutOrCall") or "").lower() in ("put", "short") else +1
        # alles in EUR rechnen: Strike kommt in Basiswertwaehrung, Preis in EUR
        kandidaten.append(strike / fx + d * px * ratio)
    return statistics.median(kandidaten) if kandidaten else None


def to_turbos(records: list[dict], spot: float | None = None,
              require_offer: bool = True, verbose: bool = True,
              fx_rate: float | None = None):
    """Wandelt SG-Suchtreffer in bewertbare Turbo-Objekte des Optimizers.

    Produkte ohne Briefkurs sind nicht kaufbar und fliegen raus -- im
    Beispielexport war das jedes achte Produkt.
    """
    from .optimizer import Turbo

    fx = fx_to_eur(records, fx_rate)
    if spot is None:
        spot = infer_spot(records, fx)
    out, verworfen = [], {}

    def drop(reason):
        verworfen[reason] = verworfen.get(reason, 0) + 1

    def zahl(v):
        """NaN ist tueckisch: `nan > 0` und `nan <= 0` sind beide False, ein
        fehlender Kurs rutscht sonst als gueltig durch und faellt erst viel
        spaeter stillschweigend heraus."""
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        return 0.0 if f != f else f

    for r in records:
        kind = SPIELRELEVANT.get(r.get("ProductClassificationId"))
        if kind is None:
            drop("Klassifikation im Spiel nicht nutzbar")
            continue
        offer = zahl(r.get("Offer"))
        bid = zahl(r.get("Bid"))
        if require_offer and offer <= 0:
            drop("kein Briefkurs (nicht kaufbar)")
            continue
        ask = offer if offer > 0 else bid
        if ask <= 0:
            drop("kein Kurs")
            continue

        # SG-Ratio ist ein Divisor, das Payoff-Modell erwartet einen
        # Multiplikator -> umrechnen, sonst liegt der Wert um Faktor 10.000 daneben.
        pc = str(r.get("PutOrCall") or "").lower()
        direction = -1 if pc in ("put", "short") else +1
        # Strike und Barriere in EUR umrechnen, damit sie zum EUR-Briefkurs passen
        barrier = zahl(r.get("BarrierTurboCertificate")) or None
        strike = zahl(r.get("Strike")) or None
        if barrier is not None:
            barrier /= fx
        if strike is not None:
            strike /= fx
        ratio = zahl(r.get("Ratio")) or 1.0
        lev = zahl(r.get("CurrentLeverage")) or None

        strike_val = None
        if kind == "turbo":
            # Barriere = Ausloeseschwelle, Strike = Wertbasis. Bei BEST Turbos
            # identisch, bei Unlimited Turbos (Mini) nicht.
            ko = barrier if barrier not in (None, 0) else strike
            if ko in (None, 0) or not spot:
                drop("Barriere oder Basiswertkurs fehlt")
                continue
            strike_val = strike if strike not in (None, 0) else ko
        elif kind == "warrant":
            ko = strike
            if ko in (None, 0) or not spot:
                drop("Basispreis oder Basiswertkurs fehlt")
                continue
        else:                                   # factor
            if not lev:
                drop("Faktor-Produkt ohne Hebelangabe")
                continue
            ko = 0.0

        out.append(Turbo(
            wkn=r.get("Code", "?"), name=f"{r.get('AssetName','?')} {pc or kind}",
            underlying=r.get("AssetName", "?"), direction=direction,
            ask=float(ask), bid=float(bid) if bid > 0 else float(ask) * 0.99,
            underlying_price=float(spot or 0.0), ko_barrier=float(ko),
            ratio=1.0 / float(ratio) if ratio else 1.0,
            leverage=float(lev) if lev else None,
            product_type=kind,
            strike=float(strike_val) if strike_val else None,
        ))

    if verbose:
        fx_hinweis = f", FX {fx:.4f}" if abs(fx - 1.0) > 1e-9 else ""
        print(f"  {len(records):,} Treffer -> {len(out):,} bewertbar"
              f"{f' (Spot ~{spot:,.2f} EUR{fx_hinweis})' if spot else ''}")
        for k, v in sorted(verworfen.items(), key=lambda x: -x[1]):
            print(f"      {v:6,}  {k}")
    return out


# Bid/Offer fehlen in der Trefferliste der Faktor-Klassifikation vollstaendig
# und stehen nur in den Detaildaten -- ohne sie ist kein Produkt kaufbar.
FACTOR_PROPS = ("FactorLeverage", "StrategyExtended", "ManagementFee",
                "AdaptionEvent", "BarrierAbsoluteLevel", "Bid", "Offer")


def enrich_factors(client: "SGClient", records: list[dict], workers: int = 6,
                   verbose: bool = True) -> list[dict]:
    """Ergaenzt Faktor-Produkte um Hebel und Richtung.

    Die Trefferliste liefert fuer Faktor-Optionsscheine weder Hebel noch
    Richtung -- beides steht nur in Products/AllProperties/{id} (Felder
    FactorLeverage und StrategyExtended). Das kostet einen Request je Produkt.
    Seriell dauert das bei mehreren hundert Produkten und der Latenz einer
    einzelnen Verbindung zu lange, deshalb ein kleiner Thread-Pool mit eigener
    Session je Thread. Die Gesamtlast bleibt gleich: die Drosselung je Session
    wird um den Parallelitaetsgrad gestreckt.
    """
    from concurrent.futures import ThreadPoolExecutor

    ziel = [r for r in records
            if SPIELRELEVANT.get(r.get("ProductClassificationId")) == "factor"]
    if not ziel:
        return records
    if verbose:
        print(f"  Reichere {len(ziel):,} Faktor-Produkte an ({workers} parallel) ...")

    clients = [SGClient(pause=client.pause * workers) for _ in range(workers)]

    def hole(idx_rec):
        idx, r = idx_rec
        try:
            return r, clients[idx % workers].properties_dict(r["Id"])
        except SGApiError:
            return r, None

    fertig = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for r, props in pool.map(hole, enumerate(ziel)):
            fertig += 1
            if verbose and fertig % 200 == 0:
                print(f"    {fertig:,}/{len(ziel):,}")
            if not props:
                continue
            for k in FACTOR_PROPS:
                if k in props:
                    r[k] = props[k]
            # FactorLeverage traegt die Richtung im Vorzeichen (-9 = Short 9x).
            # Der Betrag gehoert in die Hebelangabe, das Vorzeichen in die
            # Richtung -- sonst wird omega negativ und die Vorauswahl bricht.
            lev = None
            try:
                lev = float(props.get("FactorLeverage"))
            except (TypeError, ValueError):
                pass
            strat = str(props.get("StrategyExtended") or "").lower()
            if lev is not None:
                r["CurrentLeverage"] = abs(lev)
                r["PutOrCall"] = "put" if lev < 0 else "call"
            else:
                r["PutOrCall"] = "put" if "short" in strat else "call"
            if lev is not None and (lev < 0) != ("short" in strat) and strat:
                print(f"    [!] {r.get('Code')}: Vorzeichen {lev} widerspricht "
                      f"Strategie '{strat}'")
    return records


def validate_pricing(turbos: list, spot: float, toleranz: float = 0.15,
                     verbose: bool = True) -> tuple[list, dict]:
    """Kreuzprobe: Laesst sich der Briefkurs aus Strike und Spot nachrechnen?

    Fuer einen Turbo muss gelten  Preis ~ |Spot - Strike| * Ratio. Weicht das
    stark ab, stimmt eine Konvention nicht (falsches Bezugsverhaeltnis,
    Quanto-Produkt, veralteter Basiswertkurs). Solche Produkte werden
    verworfen, statt das Ergebnis still zu verfaelschen -- eine um Faktor 10
    falsche Ratio erzeugt sonst scheinbar risikolose Vervielfacher.
    """
    ok, raus, faktoren = [], 0, []
    for t in turbos:
        if t.product_type != "turbo":
            ok.append(t)
            continue
        strike = t.strike if t.strike is not None else t.ko_barrier
        theo = abs(spot - strike) * t.ratio
        if theo <= 0 or t.ask <= 0:
            raus += 1
            continue
        f = t.ask / theo
        faktoren.append(f)
        if abs(f - 1.0) <= toleranz:
            ok.append(t)
        else:
            raus += 1
    import statistics
    diag = {"geprueft": len(turbos), "behalten": len(ok), "verworfen": raus,
            "median_faktor": statistics.median(faktoren) if faktoren else None}
    if verbose:
        mf = diag["median_faktor"]
        print(f"  Preisprobe: {len(ok):,}/{len(turbos):,} plausibel"
              f"{f', Median Brief/Theorie {mf:.3f}' if mf else ''}"
              f"{f', {raus:,} verworfen' if raus else ''}")
    return ok, diag
