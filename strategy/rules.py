"""Die Spielregeln von Trader 2026 als ausfuehrbare Nebenbedingungen.

Jede Zahl hier stammt direkt aus dem Regelwerk der Societe Generale. Der Rest des
Repos optimiert *innerhalb* dieser Grenzen -- nichts hier wird jemals umgangen.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

# ---------------------------------------------------------------- Rahmendaten
START_CAPITAL = 100_000.0          # EUR je Depot
N_DEPOTS = 2
GAME_START = date(2026, 9, 7)      # Montag
GAME_END = date(2026, 10, 30)      # Freitag
N_WEEKS = 8

# ------------------------------------------------------------- Handelsgrenzen
MAX_WEIGHT_PER_SECURITY = 0.20     # 20 % des Depotwerts je Wertpapier
MAX_LEVERAGE_WEIGHT = 0.20         # 20 % des Depotwerts in gehebelte Derivate
MAX_LEVERAGE_EUR = 20_000.0        # absolut, unabhaengig vom Depotstand
MAX_LEVERAGE_UNITS = 20_000        # Stueck je gehebeltem Derivat
MAX_BUY_ORDERS_PER_DAY = 20        # Verkaeufe sind unbegrenzt
HOLDING_PERIOD_MINUTES = 5
MIN_STOCK_PRICE = 1.00             # EUR, zum Kaufzeitpunkt

# -------------------------------------------------------------------- Kosten
FEE_STOCK = 10.00                  # EUR je Ausfuehrung
FEE_DERIVATIVE = 3.90              # EUR je Ausfuehrung

# --------------------------------------------------------------- Handelszeit
TRADING_OPEN = "08:00"
TRADING_CLOSE = "22:00"

# ------------------------------------------------------------------ Ruecksetzer
RESETS_PER_WEEK_PER_DEPOT = 1


def leverage_budget(depot_value: float) -> float:
    """Maximal in gehebelte Derivate investierbarer Betrag zum Kaufzeitpunkt.

    Beide Grenzen greifen gleichzeitig; ab einem Depotwert von 100.000 EUR ist
    die absolute 20.000-EUR-Kappe bindend und waechst *nicht* mit dem Depot mit.
    Das macht die Hebelquote zu einer knappen, nicht regenerierbaren Ressource.
    """
    return min(MAX_LEVERAGE_WEIGHT * depot_value, MAX_LEVERAGE_EUR)


def max_position_eur(depot_value: float) -> float:
    """20-%-Grenze je einzelnem Wertpapier."""
    return MAX_WEIGHT_PER_SECURITY * depot_value


def max_units_affordable(price: float, depot_value: float, leveraged: bool) -> int:
    """Wie viele Stueck eines Papiers duerfen gekauft werden?

    Bei gehebelten Derivaten binden drei Grenzen gleichzeitig: EUR-Budget,
    20-%-Positionsgrenze und die 20.000-Stueck-Regel. Aus dem Zusammenspiel von
    20.000 EUR und 20.000 Stueck folgt der optimale Preispunkt von ~1,00 EUR
    je Schein, wenn nur ein einziges Produkt gekauft wird.
    """
    if price <= 0:
        return 0
    cap_eur = min(max_position_eur(depot_value),
                  leverage_budget(depot_value) if leveraged else float("inf"))
    units = int(cap_eur // price)
    if leveraged:
        units = min(units, MAX_LEVERAGE_UNITS)
    return max(units, 0)


def optimal_leverage_price_point(n_products: int = 1) -> float:
    """Preis je Schein, bei dem EUR- und Stueck-Kappe gleichzeitig ausgereizt sind.

    Mit einem Produkt sind das 1,00 EUR. Verteilt man das Budget auf n Produkte
    -- sofern die Stueck-Regel je Wertpapier gilt, was am ersten Spieltag zu
    verifizieren ist -- sinkt der optimale Preis auf 1/n EUR und die
    Konvexitaet des Korbes steigt entsprechend.
    """
    return MAX_LEVERAGE_EUR / (MAX_LEVERAGE_UNITS * max(n_products, 1))


def trading_days(start: date = GAME_START, end: date = GAME_END) -> list[date]:
    """Handelstage des Spielzeitraums (Mo-Fr, ohne deutsche Feiertage im Fenster).

    Im Fenster 07.09.-30.10.2026 liegt der 03.10. (Tag der Deutschen Einheit)
    auf einem Samstag, die Boerse Stuttgart handelt an allen uebrigen Werktagen.
    """
    days, cur = [], start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def game_weeks(start: date = GAME_START, end: date = GAME_END) -> list[tuple[date, date]]:
    """Spielwochen (Montag-Sonntag). Die Wochenwertung wird Sonntagnacht gezogen."""
    weeks, cur = [], start - timedelta(days=start.weekday())
    while cur <= end:
        weeks.append((cur, cur + timedelta(days=6)))
        cur += timedelta(days=7)
    return weeks


@dataclass
class OrderCheck:
    ok: bool
    units: int
    cost: float
    reason: str = ""


def check_buy(depot_value: float, cash: float, price: float, units: int,
              leveraged: bool, existing_leverage_eur: float = 0.0,
              buy_orders_used_today: int = 0) -> OrderCheck:
    """Prueft eine Kauforder gegen alle Regeln und reduziert sie notfalls.

    Entspricht dem Verhalten des Spiels: 'Sollte die Order [...] nicht direkt
    ausgefuehrt werden koennen und ist die Regelkonformitaet zum Zeitpunkt der
    Ausfuehrung nicht mehr gegeben, so wird die Order entsprechend reduziert.'
    """
    fee = FEE_DERIVATIVE if leveraged else FEE_STOCK
    if buy_orders_used_today >= MAX_BUY_ORDERS_PER_DAY:
        return OrderCheck(False, 0, 0.0, "Tageslimit von 20 Kauforders erreicht")
    if not leveraged and price < MIN_STOCK_PRICE:
        return OrderCheck(False, 0, 0.0, f"Aktienkurs {price:.2f} EUR < 1,00 EUR Mindestkurs")

    limit = max_position_eur(depot_value)
    if leveraged:
        limit = min(limit, leverage_budget(depot_value) - existing_leverage_eur)
        units = min(units, MAX_LEVERAGE_UNITS)
    units = min(units, int(limit // price) if price > 0 else 0)
    units = min(units, int(max(cash - fee, 0) // price) if price > 0 else 0)

    if units <= 0:
        return OrderCheck(False, 0, 0.0, "Order durch Regelgrenzen auf 0 reduziert")
    return OrderCheck(True, units, units * price + fee)
