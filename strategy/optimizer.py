"""Konvexitaets-Optimizer fuer den Hebel-Sleeve.

Aufgabe: Unter den drei gleichzeitig bindenden Regelgrenzen
  (a) 20 % des Depotwerts in gehebelte Derivate,
  (b) absolut 20.000 EUR,
  (c) 20.000 Stueck je Produkt,
den Korb aus Knock-out-Turbos / Optionsscheinen waehlen, der die
Gewinnwahrscheinlichkeit maximiert -- nicht den Erwartungswert.

Der entscheidende, nicht offensichtliche Effekt: Grenze (b) und (c) zusammen
erzwingen einen Mindestpreis je Schein. Mit einem einzigen Produkt sind 20.000
EUR nur bei 1,00 EUR/Schein ausschoepfbar; ein Turbo zu 0,25 EUR nimmt nur 5.000
EUR auf. Verteilt man dieselben 20.000 EUR auf vier Produkte a 0,25 EUR, ist das
Budget voll investiert -- bei vierfacher Konvexitaet. Ob (c) je Wertpapier oder
je Depot gilt, ist am ersten Spieltag mit zwei Kleinstorders zu verifizieren;
`units_cap_is_per_product` schaltet zwischen beiden Lesarten um.
"""

from __future__ import annotations

import csv
import math
import zlib
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from . import rules

TRADING_DAYS_PER_YEAR = 252


@dataclass
class Turbo:
    """Ein gehebeltes Produkt. Felder entsprechen dem SG-Produktfinder-Export."""
    wkn: str
    name: str
    underlying: str
    direction: int            # +1 Long / Call, -1 Short / Put
    ask: float                # Briefkurs in EUR (Regelgrundlage fuer alle Kappen)
    bid: float
    underlying_price: float
    ko_barrier: float
    ratio: float = 1.0        # Bezugsverhaeltnis
    leverage: float | None = None
    product_type: str = "turbo"   # turbo | factor | warrant
    days_to_expiry: int | None = None   # nur fuer Standard-Optionsscheine
    # Bei Unlimited Turbos (Mini) liegt die Stop-Loss-Schwelle VOR dem
    # Basispreis: der Wert bemisst sich am Basispreis, ausgeknockt wird an
    # der Schwelle, und bei Ausloesung erhaelt man den Restwert dazwischen.
    # Bei BEST Turbos fallen beide zusammen, dann ist der Restwert null.
    strike: float | None = None

    @property
    def spread_pct(self) -> float:
        if self.ask <= 0:
            return 1.0
        return max((self.ask - self.bid) / self.ask, 0.0)

    @property
    def omega(self) -> float:
        """Effektiver Hebel. Beim Faktor-Zertifikat konstant, beim Turbo
        Basiswert / Abstand zur Barriere."""
        if self.leverage:
            return self.leverage
        if self.product_type == "factor":
            return 1.0
        dist = abs(self.underlying_price - self.ko_barrier)
        return self.underlying_price / dist if dist > 1e-9 else float("inf")

    @property
    def distance_to_ko_pct(self) -> float:
        return abs(self.underlying_price - self.ko_barrier) / self.underlying_price

    def max_units(self, depot_value: float, units_cap: int) -> int:
        cap_eur = min(rules.max_position_eur(depot_value), rules.leverage_budget(depot_value))
        return int(min(cap_eur // self.ask, units_cap)) if self.ask > 0 else 0


# ------------------------------------------------------------------- Pfad-Sim
def simulate_underlying(spot: float, annual_vol: float, horizon_days: int,
                        n_paths: int, steps_per_day: int = 8, drift: float = 0.0,
                        t_df: float = 4.0, seed: int = 11) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pfade des Basiswerts. Gibt (Endkurs, Pfadminimum, Pfadmaximum) zurueck.

    Minimum/Maximum werden gebraucht, weil Knock-outs *intraday* ausgeloest
    werden -- eine reine Endkursbetrachtung unterschaetzt das Ausfallrisiko
    massiv und laesst enge Turbos viel zu gut aussehen.
    """
    rng = np.random.default_rng(seed)
    n_steps = horizon_days * steps_per_day
    dt = 1.0 / (TRADING_DAYS_PER_YEAR * steps_per_day)
    z = rng.standard_t(t_df, size=(n_paths, n_steps)) / np.sqrt(t_df / (t_df - 2.0))
    r = drift * dt + annual_vol * np.sqrt(dt) * z
    paths = spot * np.cumprod(1.0 + r, axis=1)
    return paths[:, -1], paths.min(axis=1), paths.max(axis=1)


def simulate_underlying_paths(spot: float, annual_vol: float, horizon_days: int,
                              n_paths: int, steps_per_day: int = 8, drift: float = 0.0,
                              t_df: float = 4.0, seed: int = 11) -> np.ndarray:
    """Wie simulate_underlying, gibt aber die vollen Pfade zurueck.

    Wird fuer Faktor-Zertifikate gebraucht, deren Wert pfadabhaengig ist.
    """
    rng = np.random.default_rng(seed)
    n_steps = horizon_days * steps_per_day
    dt = 1.0 / (TRADING_DAYS_PER_YEAR * steps_per_day)
    z = rng.standard_t(t_df, size=(n_paths, n_steps)) / np.sqrt(t_df / (t_df - 2.0))
    r = drift * dt + annual_vol * np.sqrt(dt) * z
    return spot * np.cumprod(1.0 + r, axis=1)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standardnormalverteilung ohne scipy-Abhaengigkeit."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def black_scholes(S: np.ndarray, K: float, vol: float, t_years: float,
                  direction: int = +1, r: float = 0.0) -> np.ndarray:
    """Wert eines Standard-Optionsscheins am Bewertungshorizont.

    Anders als beim Turbo ist der Schein am Horizont noch nicht faellig -- er
    hat Restzeitwert. Ihn mit dem inneren Wert zu bewerten waere grob falsch
    und wuerde aus dem Geld liegende Scheine systematisch zu schlecht rechnen.
    """
    S = np.asarray(S, dtype=float)
    if t_years <= 1e-9 or vol <= 1e-9:
        return np.maximum(direction * (S - K), 0.0)
    sq = vol * math.sqrt(t_years)
    d1 = (np.log(np.maximum(S, 1e-12) / K) + (r + 0.5 * vol * vol) * t_years) / sq
    d2 = d1 - sq
    disc = math.exp(-r * t_years)
    if direction > 0:
        return S * _norm_cdf(d1) - K * disc * _norm_cdf(d2)
    return K * disc * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def payoff_per_unit(t: Turbo, S_end: np.ndarray, S_min: np.ndarray,
                    S_max: np.ndarray, paths: np.ndarray | None = None,
                    vol: float = 0.25, horizon_days: int = 5) -> np.ndarray:
    """Wert eines Scheins am Horizont.

    Turbo: intrinsischer Wert mit Knock-out-Absorption (Barriere intraday).
    Faktor-Zertifikat: taeglicher Reset auf konstanten Hebel. Kein Knock-out,
    aber Volatilitaetsdrag -- in einem Trend zinst es auf, in einer Seitwaerts-
    bewegung zerfaellt es. Braucht den vollen Pfad, nicht nur den Endkurs.
    """
    if t.product_type == "warrant":
        # Restlaufzeit am Horizont; ohne Laufzeitangabe konservativ 60 Tage
        rest = max((t.days_to_expiry or 60) - horizon_days, 0) / 252.0
        return black_scholes(S_end, t.ko_barrier, vol, rest, t.direction) * t.ratio

    if t.product_type == "factor":
        if paths is None:
            raise ValueError("Faktor-Zertifikate brauchen den vollen Pfad "
                             "(paths=... an payoff_per_unit uebergeben)")
        r = np.diff(paths, axis=1, prepend=t.underlying_price) / np.concatenate(
            [np.full((paths.shape[0], 1), t.underlying_price), paths[:, :-1]], axis=1)
        lev = t.leverage or 1.0
        # Emittenten setzen eine Reset-Schwelle, damit der Wert nie negativ wird
        step = np.maximum(1.0 + t.direction * lev * r, 0.0)
        return t.ask * np.prod(step, axis=1)

    strike = t.strike if t.strike is not None else t.ko_barrier
    if t.direction > 0:
        knocked = S_min <= t.ko_barrier
        intrinsic = np.maximum(S_end - strike, 0.0)
        residual = max(t.ko_barrier - strike, 0.0)      # Mini: Restwert bei Stop-Loss
    else:
        knocked = S_max >= t.ko_barrier
        intrinsic = np.maximum(strike - S_end, 0.0)
        residual = max(strike - t.ko_barrier, 0.0)
    return np.where(knocked, residual * t.ratio, intrinsic * t.ratio)


# --------------------------------------------------------------- Vorauswahl
def preselect_candidates(products: list[Turbo], max_products: int = 400,
                         bucket_step: float = 1.35, per_bucket: int = 2,
                         verbose: bool = True) -> list[Turbo]:
    """Reduziert ein grosses Produktuniversum auf einen repraesentativen Korb.

    Die SG-API liefert je Basiswert mehrere tausend Turbos. Eine Payoff-Matrix
    ueber 10.000 Produkte x 30.000 Pfade waere rund 2,4 GB gross -- und
    voellig unnoetig: Zwei Turbos mit gleichem Hebel, gleicher Richtung und
    gleichem Basiswert sind wirtschaftlich identisch, sie unterscheiden sich
    nur in Preis und Spread.

    Deshalb wird je (Basiswert, Richtung, Produkttyp) ein logarithmisches
    Hebelraster gebildet und darin jeweils der Schein mit dem engsten Spread
    behalten. Ein zweiter je Feld bleibt als Preisalternative erhalten, damit
    sich das 20.000-EUR-Budget trotz der 20.000-Stueck-Kappe ausschoepfen
    laesst.
    """
    gruppen: dict[tuple, dict[int, list[Turbo]]] = {}
    for p in products:
        o = p.omega
        if not math.isfinite(o) or o <= 0 or p.ask <= 0:
            continue
        key = (p.underlying, p.direction, p.product_type)
        b = int(round(math.log(o) / math.log(bucket_step)))
        gruppen.setdefault(key, {}).setdefault(b, []).append(p)

    felder: list[list[Turbo]] = []
    for buckets in gruppen.values():
        for bp in buckets.values():
            # enger Spread zuerst; bei Gleichstand der teurere Schein, weil er
            # unter der Stueck-Kappe mehr Budget aufnimmt
            bp.sort(key=lambda x: (round(x.spread_pct, 4), -x.ask))
            felder.append(bp[:per_bucket])

    # Round-Robin statt Abschneiden, damit keine Hebelregion wegfaellt
    out: list[Turbo] = []
    i = 0
    while len(out) < max_products and any(len(f) > i for f in felder):
        for f in felder:
            if i < len(f):
                out.append(f[i])
                if len(out) >= max_products:
                    break
        i += 1

    if verbose:
        print(f"  Vorauswahl: {len(products):,} -> {len(out)} Produkte "
              f"({len(gruppen)} Gruppen, {len(felder)} Hebelfelder)")
    return out


# ------------------------------------------------------------------ Optimizer
@dataclass
class Basket:
    units: dict[str, int]
    cost: float
    products: list[Turbo]
    objective_value: float
    stats: dict

    def to_table(self) -> str:
        lines = [f"{'WKN':12s} {'Produkt':26s} {'Brief':>7s} {'Stueck':>7s} "
                 f"{'Einsatz':>9s} {'Hebel':>6s} {'KO-Abst':>8s}"]
        lines.append("-" * 82)
        for p in self.products:
            u = self.units.get(p.wkn, 0)
            if u == 0:
                continue
            lines.append(f"{p.wkn:12s} {p.name[:26]:26s} {p.ask:7.3f} {u:7d} "
                         f"{u * p.ask:9,.0f} {p.omega:6.1f} {p.distance_to_ko_pct * 100:7.1f}%")
        lines.append("-" * 82)
        lines.append(f"{'Summe':12s} {'':26s} {'':>7s} {'':>7s} {self.cost:9,.0f}")
        return "\n".join(lines)


def optimise_basket(products: list[Turbo], depot_value: float,
                    horizon_days: int, vol_by_underlying: dict[str, float],
                    objective: str = "p_target", target_multiple: float = 2.5,
                    alpha: float = 2.0, rest_of_depot: float | None = None,
                    n_paths: int = 40_000, units_cap_is_per_product: bool = True,
                    n_restarts: int = 400, seed: int = 11,
                    max_candidates: int = 400) -> Basket:
    """Waehlt Stueckzahlen je Produkt unter allen Regelgrenzen.

    objective:
      "p_target" -- maximiert P(Depot >= target_multiple * Startwert). Das ist
                    die Zielfunktion, die zum Spiel passt: es gibt keinen Preis
                    fuer Platz 300, also zaehlt nur die rechte Verteilungsschulter.
      "power"    -- maximiert E[Depot^alpha], eine glattere risikofreudige Variante.
    """
    if not products:
        raise ValueError("keine Produkte uebergeben")
    # Doppelte WKN wuerden das Ergebnis-Dict kollabieren lassen: to_table()
    # zeigt dann leere Koerbe an, obwohl Budget eingesetzt wurde.
    eindeutig, gesehen = [], set()
    for p in products:
        if p.wkn not in gesehen:
            gesehen.add(p.wkn)
            eindeutig.append(p)
    if len(eindeutig) < len(products):
        print(f"  [optimizer] {len(products) - len(eindeutig)} doppelte WKN entfernt")
    products = eindeutig
    if len(products) > max_candidates:
        products = preselect_candidates(products, max_candidates)

    budget = rules.leverage_budget(depot_value)
    pos_cap = rules.max_position_eur(depot_value)
    units_cap = rules.MAX_LEVERAGE_UNITS
    if rest_of_depot is None:
        rest_of_depot = depot_value - budget          # Aktien-Sleeve + Cash

    # Pfade je Basiswert einmal ziehen, damit Produkte auf demselben Underlying
    # korrekt korreliert bewertet werden.
    payoffs = np.zeros((n_paths, len(products)))
    needs_paths = any(p.product_type == "factor" for p in products)
    for i, p in enumerate(products):
        vol = vol_by_underlying.get(p.underlying, 0.25)
        # zlib.crc32 statt hash(): Pythons String-Hash ist je Prozess
        # randomisiert, damit waeren die Studienergebnisse nicht reproduzierbar.
        sd = seed + zlib.crc32(p.underlying.encode()) % 9973
        if needs_paths:
            paths = simulate_underlying_paths(p.underlying_price, vol, horizon_days,
                                              n_paths, seed=sd)
            S_end, S_min, S_max = paths[:, -1], paths.min(axis=1), paths.max(axis=1)
        else:
            paths = None
            S_end, S_min, S_max = simulate_underlying(
                p.underlying_price, vol, horizon_days, n_paths, seed=sd)
        payoffs[:, i] = payoff_per_unit(p, S_end, S_min, S_max, paths,
                                        vol=vol, horizon_days=horizon_days)

    ask = np.array([p.ask for p in products])
    spread = np.array([p.spread_pct for p in products])
    # Einstiegsspread ist sofort verloren: effektiver Einstandswert je Schein
    eff = payoffs * (1.0 - spread)[None, :]

    target = target_multiple * depot_value
    max_units_vec = np.minimum(pos_cap / np.maximum(ask, 1e-9),
                               units_cap if units_cap_is_per_product else units_cap)

    def score(units: np.ndarray) -> float:
        sleeve = eff @ units
        final = rest_of_depot + sleeve
        if objective == "p_target":
            return float(np.mean(final >= target))
        return float(np.mean(np.power(np.maximum(final, 1.0) / depot_value, alpha)))

    rng = np.random.default_rng(seed)
    best_units = np.zeros(len(products))
    best = -np.inf

    def project(w: np.ndarray) -> np.ndarray:
        """Zufaellige Gewichte -> regelkonforme, ganzzahlige Stueckzahlen."""
        w = np.maximum(w, 0.0)
        if w.sum() <= 0:
            return np.zeros(len(products))
        spend = budget * w / w.sum()
        u = np.floor(np.minimum(spend / np.maximum(ask, 1e-9), max_units_vec))
        # Gesamtbudget nach dem Abrunden ggf. nachjustieren
        while (u * ask).sum() > budget and u.sum() > 0:
            j = int(np.argmax(u * ask))
            u[j] -= 1
        if not units_cap_is_per_product and u.sum() > units_cap:
            scale = units_cap / u.sum()
            u = np.floor(u * scale)
        return u

    n = len(products)

    # Strukturierte Startpunkte. Der entscheidende Kandidatentyp ist der
    # richtungs- und hebelreine Korb: ein einzelnes Produkt nimmt wegen der
    # 20.000-Stueck-Kappe nur `20.000 * Preis` Euro auf, also fuellt man den
    # Rest mit Produkten gleicher Richtung und gleichem Hebel auf. Genau das
    # ist die praktische Handlungsregel -- und ohne diese Startpunkte findet
    # eine rein zufaellige Suche stattdessen gemischte Long/Short-Koerbe,
    # die die Konvexitaet zerstoeren.
    structured: list[np.ndarray] = []
    families: dict[tuple, list[int]] = {}
    for i, p in enumerate(products):
        families.setdefault((p.direction, round(p.omega, 2)), []).append(i)
    for idxs in families.values():
        u = np.zeros(n)
        left = budget
        for i in sorted(idxs, key=lambda j: -ask[j]):     # teure zuerst: mehr Euro je Stueck
            if left <= 0:
                break
            take = min(np.floor(left / ask[i]), max_units_vec[i])
            if take <= 0:
                continue
            u[i] = take
            left -= take * ask[i]
        structured.append(u)
    # zusaetzlich: richtungsreine Koerbe ueber mehrere Hebel derselben Richtung
    for d in (+1, -1):
        idxs = [i for i, p in enumerate(products) if p.direction == d]
        if idxs:
            u = np.zeros(n)
            left = budget
            for i in sorted(idxs, key=lambda j: (-products[j].omega, -ask[j])):
                if left <= 0:
                    break
                take = min(np.floor(left / ask[i]), max_units_vec[i])
                if take > 0:
                    u[i] = take
                    left -= take * ask[i]
            structured.append(u)

    for u in structured:
        if not units_cap_is_per_product and u.sum() > units_cap:
            u = np.floor(u * (units_cap / u.sum()))
        s = score(u)
        if s > best:
            best, best_units = s, u

    for k in range(n_restarts):
        if k < n:                                   # jedes Produkt einmal solo
            w = np.zeros(n); w[k % n] = 1.0
        elif k % 5 == 0:                            # sparsame Koerbe
            w = np.zeros(n)
            for j in rng.choice(n, size=min(rng.integers(2, 5), n), replace=False):
                w[j] = rng.random()
        else:
            w = rng.dirichlet(np.full(n, 0.35))     # konzentriert, nicht gleichverteilt
        u = project(w)
        s = score(u)
        if s > best:
            best, best_units = s, u

    # lokale Verfeinerung: Stueck einzeln umschichten
    step = max(int(0.05 * units_cap), 50)
    while step >= 10:
        improved = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                u = best_units.copy()
                if u[i] < step:
                    continue
                move_cost = step * ask[i]
                add = min(np.floor(move_cost / ask[j]), max_units_vec[j] - u[j])
                if add <= 0:
                    continue
                u[i] -= step; u[j] += add
                if (u * ask).sum() > budget:
                    continue
                s = score(u)
                if s > best:
                    best, best_units, improved = s, u, True
        if not improved:
            step //= 2

    sleeve = eff @ best_units
    final = rest_of_depot + sleeve
    stats = {
        "budget_eur": budget,
        "eingesetzt_eur": float((best_units * ask).sum()),
        "budget_ausgeschoepft_%": float((best_units * ask).sum() / budget * 100),
        "P(Totalverlust Sleeve)": float(np.mean(sleeve <= 1.0)),
        "Sleeve Median-Faktor": float(np.median(sleeve) / max((best_units * ask).sum(), 1)),
        "Sleeve p95-Faktor": float(np.percentile(sleeve, 95) / max((best_units * ask).sum(), 1)),
        "Sleeve p99-Faktor": float(np.percentile(sleeve, 99) / max((best_units * ask).sum(), 1)),
        f"P(Depot >= {target_multiple}x)": float(np.mean(final >= target)),
        "Depot Median": float(np.median(final)),
        "Depot p99": float(np.percentile(final, 99)),
    }
    return Basket({p.wkn: int(u) for p, u in zip(products, best_units)},
                  float((best_units * ask).sum()), products, best, stats)


# ------------------------------------------------------------------ CSV-Import
CSV_ALIASES = {
    "wkn": ["wkn", "wertpapierkennnummer"],
    "name": ["name", "produktname", "bezeichnung", "produkt"],
    "underlying": ["underlying", "basiswert", "bw"],
    "ask": ["ask", "brief", "briefkurs"],
    "bid": ["bid", "geld", "geldkurs"],
    "underlying_price": ["underlying_price", "basiswertkurs", "kurs basiswert", "spot"],
    "ko_barrier": ["ko_barrier", "knock-out", "knock out", "ko-schwelle", "barriere", "basispreis", "strike"],
    "ratio": ["ratio", "bezugsverhaeltnis", "bezugsverhältnis", "bv"],
    "leverage": ["leverage", "hebel", "omega"],
    "direction": ["direction", "typ", "type", "richtung"],
}


def load_products_csv(path: str | Path) -> list[Turbo]:
    """Liest einen Export aus dem SG-Produktfinder (sg-zertifikate.de).

    Spaltennamen werden tolerant gemappt; Dezimalkomma wird akzeptiert.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8-sig")
    delim = ";" if raw.count(";") > raw.count(",") else ","
    rows = list(csv.DictReader(raw.splitlines(), delimiter=delim))
    if not rows:
        return []

    header = {h.strip().lower(): h for h in rows[0].keys() if h}
    colmap = {}
    for field_name, aliases in CSV_ALIASES.items():
        for a in aliases:
            for h_low, h_orig in header.items():
                if a == h_low or a in h_low:
                    colmap[field_name] = h_orig
                    break
            if field_name in colmap:
                break

    def num(row, key, default=None):
        col = colmap.get(key)
        if not col or row.get(col) in (None, "", "-"):
            return default
        s = str(row[col]).replace(".", "").replace(",", ".") if "," in str(row[col]) else str(row[col])
        try:
            return float(s.replace(" ", "").replace("%", ""))
        except ValueError:
            return default

    out: list[Turbo] = []
    for row in rows:
        wkn = (row.get(colmap.get("wkn", ""), "") or "").strip()
        if not wkn:
            continue
        name = (row.get(colmap.get("name", ""), "") or wkn).strip()
        typ = (row.get(colmap.get("direction", ""), "") or name).lower()
        direction = -1 if any(k in typ for k in ("short", "put", "bear")) else +1
        ask = num(row, "ask")
        if not ask:
            continue
        is_factor = "faktor" in typ or "factor" in typ
        spot = num(row, "underlying_price")
        ko = num(row, "ko_barrier")
        lev = num(row, "leverage")

        if is_factor:
            if not lev:
                print(f"  [csv] {wkn}: Faktor-Produkt ohne Hebelangabe -- uebersprungen")
                continue
            ko = 0.0
        elif ko is None or spot is None:
            # Kein Phantasiewert: ein Turbo ohne Barriere oder ohne Basiswert-
            # kurs laesst sich nicht bewerten und wird lieber verworfen.
            print(f"  [csv] {wkn}: Knock-out oder Basiswertkurs fehlt -- uebersprungen")
            continue

        out.append(Turbo(
            wkn=wkn, name=name,
            underlying=(row.get(colmap.get("underlying", ""), "") or "?").strip(),
            direction=direction, ask=ask, bid=num(row, "bid", ask * 0.99),
            underlying_price=spot if spot is not None else 100.0,
            ko_barrier=ko if ko is not None else 0.0,
            ratio=num(row, "ratio", 1.0) or 1.0,
            leverage=lev,
            product_type="factor" if is_factor else "turbo",
        ))
    return out


def synthetic_universe(underlying: str, spot: float, direction: int = +1,
                       leverages=(8, 12, 18, 25, 35, 50, 70),
                       target_prices=(0.10, 0.25, 0.50, 1.00, 2.00),
                       spread_pct: float = 0.008) -> list[Turbo]:
    """Realistisches Testuniversum, solange kein SG-Export vorliegt.

    Fuer jede Kombination aus Hebel und Zielpreis wird das Bezugsverhaeltnis so
    gewaehlt, dass der Schein genau diesen Preis kostet -- damit laesst sich der
    Effekt der 20.000-Stueck-Regel sauber isolieren.
    """
    out = []
    for L in leverages:
        ko = spot * (1.0 - direction / L)
        dist = abs(spot - ko)
        for px in target_prices:
            ratio = px / dist
            out.append(Turbo(
                wkn=f"SYN{'L' if direction > 0 else 'S'}{L:02d}{int(px * 100):03d}",
                name=f"{'Long' if direction > 0 else 'Short'} x{L} @{px:.2f}",
                underlying=underlying, direction=direction,
                ask=px, bid=px * (1 - spread_pct),
                underlying_price=spot, ko_barrier=ko, ratio=ratio, leverage=float(L),
            ))
    return out


# --------------------------------------------------------- Faire Bepreisung
_FAIR_CACHE: dict[tuple, float] = {}


def fair_value_factor(leverage: float, annual_vol: float, horizon_days: int,
                      direction: int = +1, t_df: float = 4.0,
                      n_paths: int = 120_000, seed: int = 5) -> float:
    """E[Auszahlung] / Kosten eines KO-Turbos bei driftfreiem Basiswert.

    Bei fetten Raendern gappt der Basiswert durch die Barriere: der Schein zahlt
    dann 0 statt eines negativen Werts. Diese beschraenkte Haftung ist fuer den
    Kaeufer ein Vorteil -- ein naiv modellierter Turbo mit Hebel 130 wirft so
    ueber eine Woche rund 22 % Gratisrendite ab, die es real nicht gibt.
    Emittenten preisen das Gap-Risiko ueber Finanzierungsspread und
    Stop-Loss-Schwelle ein. Der hier zurueckgegebene Faktor macht genau das:
    Preis mal Faktor = arbitragefreier Einstand.
    """
    key = (round(leverage, 3), round(annual_vol, 4), horizon_days, direction, round(t_df, 2))
    if key in _FAIR_CACHE:
        return _FAIR_CACHE[key]

    spot = 100.0
    ko = spot * (1.0 - direction / leverage)
    t = Turbo("FAIR", "fair", "X", direction, abs(spot - ko), abs(spot - ko),
              spot, ko, ratio=1.0, leverage=leverage)
    S_end, S_min, S_max = simulate_underlying(spot, annual_vol, horizon_days,
                                              n_paths, t_df=t_df, seed=seed)
    factor = float(np.mean(payoff_per_unit(t, S_end, S_min, S_max)) / t.ask)
    _FAIR_CACHE[key] = factor
    return factor


def apply_fair_pricing(products: list[Turbo], vol_by_underlying: dict[str, float],
                       horizon_days: int) -> list[Turbo]:
    """Hebt die Briefkurse synthetischer Produkte auf ihren fairen Wert an.

    Nur fuer das synthetische Testuniversum noetig -- echte Marktpreise aus dem
    SG-Export enthalten das Gap-Risiko bereits.
    """
    out = []
    for p in products:
        vol = vol_by_underlying.get(p.underlying, 0.25)
        f = fair_value_factor(p.omega, vol, horizon_days, p.direction)
        out.append(replace_turbo(p, ask=p.ask * f, bid=p.bid * f))
    return out


def replace_turbo(t: Turbo, **kw) -> Turbo:
    d = asdict(t)
    d.update(kw)
    return Turbo(**d)


# ------------------------------------------------------- SG-Excel-Import
# Produktarten des SG-Produktfinders -> Modelltyp und Handelbarkeit im Spiel.
# "Classic Aktienanleihen" sind nach Ziffer 4 der Spielregeln ausdruecklich
# nicht handelbar (Stueckzinsen) und werden hart herausgefiltert.
SG_PRODUCT_TYPES = {
    "BEST Turbo-Optionsscheine (Open-End)":  ("turbo",  True),
    "Unlimited Turbo-Optionsscheine (Mini)": ("turbo",  True),
    "Classic Turbo-Optionsscheine":          ("turbo",  True),
    "Faktor-Optionsscheine":                 ("factor", True),
    "Standard-Optionsscheine (Call)":        ("warrant", True),
    "Standard-Optionsscheine (Put)":         ("warrant", True),
    "Inline-Optionsscheine":                 ("inline", True),
    "Classic Discount-Zertifikate":          ("discount", False),
    "Capped Bonus-Zertifikate":              ("bonus", False),
    "Classic Bonus-Zertifikate":             ("bonus", False),
    "Memory Express-Zertifikate":            ("express", False),
    "Fixkupon Express-Zertifikate":          ("express", False),
    "Classic Aktienanleihen":                ("aktienanleihe", False),  # im Spiel verboten
}

# Produktarten, die unter die 20-%/20.000-EUR-Hebelkappe fallen. Die
# Spielregeln nennen "Optionsscheine, Turbo-Optionsscheine und Faktor-
# Optionsscheine". Ob das Spiel Inline-Optionsscheine ebenfalls als gehebelt
# fuehrt, ist offen -- siehe Testplan im README.
SG_LEVERAGED = {"turbo", "factor", "warrant"}


def _de_num(series):
    """Deutsche Zahlenformate aus dem SG-Export in float wandeln."""
    import pandas as pd
    s = (series.astype(str)
         .str.replace(" EUR", "", regex=False).str.replace(" USD", "", regex=False)
         .str.replace(" ", "", regex=False).str.strip())
    # Punkt ist Tausendertrenner nur, wenn danach noch ein Komma kommt
    has_comma = s.str.contains(",", regex=False)
    s = s.where(~has_comma, s.str.replace(".", "", regex=False))
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def load_products_xlsx(path: str | Path, assumed_ratio: float = 1.0,
                       verbose: bool = True) -> tuple[list[Turbo], dict]:
    """Liest einen ProductSearch-Export (.xlsx) der Societe Generale.

    Gibt (bewertbare Produkte, Diagnose) zurueck. Die Diagnose benennt
    ausdruecklich, welche Felder fehlen -- der Standardexport enthaelt weder
    Knock-out-Schwelle noch Bezugsverhaeltnis noch die Long/Short-Richtung,
    und ohne diese drei laesst sich ein Turbo nicht bewerten. Es wird nichts
    geraten: Produkte ohne ausreichende Daten landen in der Diagnose, nicht
    im Ergebnis.
    """
    import pandas as pd

    df = pd.read_excel(path, header=1)
    diag: dict = {"zeilen_gesamt": len(df), "spalten": list(df.columns)}

    colmap = {c.lower().strip(): c for c in df.columns}

    def col(*names):
        for n in names:
            for low, orig in colmap.items():
                if n in low:
                    return orig
        return None

    c_wkn = col("wkn")
    c_bw = col("basiswert")
    c_art = col("produktart")
    c_spot = col("kurs basiswert")
    c_bid = col("geld")
    c_ask = col("brief")
    c_ko = col("knock", "basispreis", "strike", "barriere")
    c_ratio = col("bezugsverh", "ratio")
    c_lev = col("hebel", "omega")
    c_dir = col("richtung", "typ", "call/put", "long/short")
    c_exp = col("bewertungstag", "laufzeit", "faelligkeit")

    diag["fehlende_pflichtfelder"] = [n for n, c in
                                      [("Knock-out/Basispreis", c_ko),
                                       ("Bezugsverhaeltnis", c_ratio),
                                       ("Hebel", c_lev),
                                       ("Long/Short-Richtung", c_dir)] if c is None]

    df["_ask"] = _de_num(df[c_ask]) if c_ask else None
    df["_bid"] = _de_num(df[c_bid]) if c_bid else None
    df["_spot"] = _de_num(df[c_spot]) if c_spot else None
    df["_ko"] = _de_num(df[c_ko]) if c_ko else float("nan")
    df["_ratio"] = _de_num(df[c_ratio]) if c_ratio else float("nan")
    df["_lev"] = _de_num(df[c_lev]) if c_lev else float("nan")

    kinds, tradeable = [], []
    for art in df[c_art].fillna(""):
        k, t = SG_PRODUCT_TYPES.get(str(art).strip(), ("unbekannt", False))
        kinds.append(k)
        tradeable.append(t)
    df["_kind"] = kinds
    df["_spielbar"] = tradeable

    diag["nach_produktart"] = df[c_art].value_counts().to_dict()
    diag["im_spiel_nicht_handelbar"] = int((~df["_spielbar"]).sum())
    diag["ohne_briefkurs"] = int((df["_ask"].isna() | (df["_ask"] <= 0)).sum())
    diag["basiswerte"] = sorted(df[c_bw].dropna().unique().tolist())

    out: list[Turbo] = []
    verworfen: dict[str, int] = {}

    def drop(reason):
        verworfen[reason] = verworfen.get(reason, 0) + 1

    for _, r in df.iterrows():
        if not r["_spielbar"]:
            drop("Produktart im Spiel nicht handelbar")
            continue
        ask = r["_ask"]
        if not ask or ask <= 0:
            drop("kein Briefkurs (nicht kaufbar)")
            continue

        kind = r["_kind"]
        art = str(r[c_art])
        direction = -1 if ("Put" in art or "Short" in art) else +1
        if c_dir and isinstance(r.get(c_dir), str):
            if any(k in r[c_dir].lower() for k in ("short", "put", "bear")):
                direction = -1

        lev = r["_lev"] if r["_lev"] == r["_lev"] else None
        ratio = r["_ratio"] if r["_ratio"] == r["_ratio"] else assumed_ratio
        spot = r["_spot"]
        ko = r["_ko"]

        if kind == "factor":
            if lev is None:
                drop("Faktor-Produkt ohne Hebelangabe")
                continue
            ko = 0.0
        elif kind == "turbo":
            if ko != ko:                    # NaN -> keine Barriere im Export
                drop("Turbo ohne Knock-out-Schwelle -- nicht bewertbar")
                continue
            if spot != spot:
                drop("kein Basiswertkurs")
                continue
        elif kind == "warrant":
            if ko != ko:
                drop("Optionsschein ohne Basispreis -- nicht bewertbar")
                continue
            if spot != spot:
                drop("kein Basiswertkurs")
                continue
        else:
            drop(f"Typ '{kind}' wird vom Payoff-Modell nicht abgedeckt")
            continue

        days_exp = None
        if c_exp is not None:
            val = r.get(c_exp)
            if isinstance(val, pd.Timestamp):
                days_exp = max((val.date() - pd.Timestamp.today().date()).days, 1)

        out.append(Turbo(
            wkn=str(r[c_wkn]).strip(), name=f"{r[c_bw]} {art}",
            underlying=str(r[c_bw]).strip(), direction=direction,
            ask=float(ask), bid=float(r["_bid"]) if r["_bid"] == r["_bid"] else float(ask) * 0.99,
            underlying_price=float(spot) if spot == spot else 0.0,
            ko_barrier=float(ko) if ko == ko else 0.0,
            ratio=float(ratio), leverage=float(lev) if lev else None,
            product_type=kind, days_to_expiry=days_exp,
        ))

    diag["verworfen"] = verworfen
    diag["bewertbar"] = len(out)

    if verbose:
        print(f"[SG-Import] {len(df)} Zeilen -> {len(out)} bewertbare Produkte")
        if diag["fehlende_pflichtfelder"]:
            print("  FEHLENDE SPALTEN:", ", ".join(diag["fehlende_pflichtfelder"]))
        for reason, n in sorted(verworfen.items(), key=lambda x: -x[1]):
            print(f"    {n:5d}  {reason}")
    return out, diag


def load_products_dir(directory: str | Path, pattern: str = "*.xlsx",
                      assumed_ratio: float = 1.0) -> tuple[list[Turbo], dict]:
    """Liest alle SG-Exporte eines Verzeichnisses und fuehrt sie zusammen.

    Der Produktfinder deckelt jeden Export bei 5.000 Produkten, das Universum
    umfasst aber rund 300.000. Deshalb wird in Scheiben exportiert (je Basiswert
    und Produktart) und hier wieder zusammengesetzt; Duplikate ueber die WKN
    werden entfernt.
    """
    directory = Path(directory)
    alle: dict[str, Turbo] = {}
    gesamt = {"dateien": [], "zeilen": 0, "verworfen": {}, "basiswerte": set()}
    for f in sorted(directory.glob(pattern)):
        prods, diag = load_products_xlsx(f, assumed_ratio, verbose=False)
        for p in prods:
            alle[p.wkn] = p
        gesamt["dateien"].append({"datei": f.name, "zeilen": diag["zeilen_gesamt"],
                                  "bewertbar": diag["bewertbar"],
                                  "fehlend": diag["fehlende_pflichtfelder"]})
        gesamt["zeilen"] += diag["zeilen_gesamt"]
        gesamt["basiswerte"].update(diag["basiswerte"])
        for k, v in diag["verworfen"].items():
            gesamt["verworfen"][k] = gesamt["verworfen"].get(k, 0) + v
    gesamt["basiswerte"] = sorted(gesamt["basiswerte"])
    gesamt["bewertbar_gesamt"] = len(alle)

    print(f"[SG-Import] {len(gesamt['dateien'])} Datei(en), {gesamt['zeilen']:,} Zeilen "
          f"-> {len(alle):,} eindeutige bewertbare Produkte")
    for d in gesamt["dateien"]:
        miss = f"  FEHLT: {', '.join(d['fehlend'])}" if d["fehlend"] else ""
        print(f"    {d['datei']:<44} {d['zeilen']:>6,} -> {d['bewertbar']:>6,}{miss}")
    return list(alle.values()), gesamt
