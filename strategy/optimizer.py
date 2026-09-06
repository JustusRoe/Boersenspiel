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
    product_type: str = "turbo"   # "turbo" (Knock-out) oder "factor" (Faktor-Zertifikat)

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


def payoff_per_unit(t: Turbo, S_end: np.ndarray, S_min: np.ndarray,
                    S_max: np.ndarray, paths: np.ndarray | None = None) -> np.ndarray:
    """Wert eines Scheins am Horizont.

    Turbo: intrinsischer Wert mit Knock-out-Absorption (Barriere intraday).
    Faktor-Zertifikat: taeglicher Reset auf konstanten Hebel. Kein Knock-out,
    aber Volatilitaetsdrag -- in einem Trend zinst es auf, in einer Seitwaerts-
    bewegung zerfaellt es. Braucht den vollen Pfad, nicht nur den Endkurs.
    """
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

    if t.direction > 0:
        knocked = S_min <= t.ko_barrier
        intrinsic = np.maximum(S_end - t.ko_barrier, 0.0)
    else:
        knocked = S_max >= t.ko_barrier
        intrinsic = np.maximum(t.ko_barrier - S_end, 0.0)
    return np.where(knocked, 0.0, intrinsic * t.ratio)


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
                    n_restarts: int = 400, seed: int = 11) -> Basket:
    """Waehlt Stueckzahlen je Produkt unter allen Regelgrenzen.

    objective:
      "p_target" -- maximiert P(Depot >= target_multiple * Startwert). Das ist
                    die Zielfunktion, die zum Spiel passt: es gibt keinen Preis
                    fuer Platz 300, also zaehlt nur die rechte Verteilungsschulter.
      "power"    -- maximiert E[Depot^alpha], eine glattere risikofreudige Variante.
    """
    if not products:
        raise ValueError("keine Produkte uebergeben")

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
        payoffs[:, i] = payoff_per_unit(p, S_end, S_min, S_max, paths)

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
