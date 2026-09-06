"""Monte-Carlo-Simulation des kompletten Trader-2026-Spiels.

Die Kernfrage, die dieses Modul beantwortet: Wie aggressiv muss man spielen, um
P(Gesamtsieg) zu maximieren -- und wie viel ist die woechentliche
Ruecksetzer-Option tatsaechlich wert?

Modellierung:
  * Marktfaktor als Student-t-Prozess mit optionalem Crash-Regime.
  * Aktien-Sleeve: bis zu 5 Positionen a 20 %, Beta zum Markt plus
    idiosynkratische Vola plus Sprungkomponente (Squeeze / Gap).
  * Hebel-Sleeve: echte Knock-out-Turbos. Wert = Stueck * (S - K) * Ratio,
    Totalverlust bei Barrierenberuehrung. Die Barriere wird auf Intraday-
    Schritten geprueft, nicht nur auf Tagesschluss -- sonst wird das
    Knock-out-Risiko systematisch unterschaetzt.
  * Alle Regelgrenzen aus strategy.rules greifen beim Kauf.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import rules

TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------- Modelle
@dataclass
class MarketModel:
    """Marktfaktor. Defaults kalibriert auf die Lage im September 2026."""
    annual_vol: float = 0.16          # VIX ~14.5 -> ruhiges Regime
    annual_drift: float = 0.05
    t_df: float = 4.0                 # fette Raender
    crash_prob_daily: float = 0.008   # Midterm-Jahr-Saisonalitaet Sep-Okt
    crash_mean: float = -0.035
    crash_vol: float = 0.025


@dataclass
class StockSleeve:
    """Der ungehebelte Teil -- in der Praxis der eigentliche Renditemotor,
    weil er nicht unter die 20.000-EUR-Kappe faellt."""
    n_positions: int = 5
    weight_each: float = 0.20
    annual_vol: float = 0.55          # Vola der gewaehlten Titel
    beta: float = 1.1
    jump_prob: float = 0.012          # taegliche Wahrscheinlichkeit eines Sprungs
    jump_up: float = 0.55
    jump_down: float = -0.32
    # Standard: erwartungswertneutral (p*up + (1-p)*down = 0). Alles darueber
    # ist ein unterstellter Selektionsvorteil und muss bewusst gesetzt werden --
    # sonst schenkt sich die Simulation eine Rendite, die es nicht gibt.
    jump_up_share: float | None = None

    @property
    def neutral_up_share(self) -> float:
        return abs(self.jump_down) / (self.jump_up + abs(self.jump_down))

    @property
    def effective_up_share(self) -> float:
        return self.neutral_up_share if self.jump_up_share is None else self.jump_up_share


@dataclass
class LeverageSleeve:
    """Knock-out-Turbos unter der 20.000-EUR- und 20.000-Stueck-Kappe."""
    n_products: int = 4
    initial_leverage: float = 20.0    # Omega zum Kaufzeitpunkt
    target_price: float = 0.25        # EUR je Schein -> steuert die Stueck-Kappe
    underlying_annual_vol: float = 0.16
    underlying_beta: float = 1.0
    direction: int = +1               # +1 Long-Turbo, -1 Short-Turbo
    underlying_drift: float = 0.0     # siehe Kommentar unten
    spread_pct: float = 0.010         # Geld-Brief-Spanne, sofort verloren
    redeploy_weekly: bool = True      # freies Budget montags neu einsetzen
    fair_pricing: bool = True         # Gap-Risiko einpreisen, s. optimizer.fair_value_factor
    # Warum underlying_drift standardmaessig 0 ist: Bei einem Knock-out-Turbo
    # wird der Basispreis taeglich um Finanzierungskosten (Zins + Emittenten-
    # spread) fortgeschrieben. Diese Kosten heben den risikofreien Teil der
    # Basiswertdrift exakt auf; uebrig bleibt nur die Risikopraemie nach
    # Kosten, die aktuell nahe null liegt. Wuerde man hier die volle
    # Marktdrift ansetzen, wuerde sie mit dem Hebel multipliziert und man
    # bekaeme einen frei erfundenen Vorteil von mehreren hundert Prozent.


@dataclass
class Policy:
    name: str
    stock: StockSleeve = field(default_factory=StockSleeve)
    leverage: LeverageSleeve = field(default_factory=LeverageSleeve)
    reset_below: float = 100_000.0    # Ruecksetzer, wenn Wochenstart darunter
    use_resets: bool = True
    ride_winners: bool = True         # Gewinner nie trimmen
    # Offene Regelfrage: wird die 20-%-Hebelquote auf Marktwert oder auf
    # Anschaffungskosten geprueft? Am ersten Spieltag zu verifizieren.
    leverage_check: str = "market_value"   # oder "cost_basis"


# ----------------------------------------------------------------- Simulation
@dataclass
class SimResult:
    final_values: np.ndarray          # (N,) Depotwert am letzten Spieltag
    best_daily_return: np.ndarray     # (N,) beste Tagesperformance
    best_weekly_return: np.ndarray    # (N,) beste Wochenperformance
    n_resets: np.ndarray              # (N,) genutzte Ruecksetzer
    daily_values: np.ndarray          # (N, T+1) Wertverlauf

    def p_above(self, threshold: float) -> float:
        return float(np.mean(self.final_values >= threshold))

    def summary(self) -> dict:
        fv = self.final_values
        return {
            "median": float(np.median(fv)),
            "mean": float(np.mean(fv)),
            "p10": float(np.percentile(fv, 10)),
            "p90": float(np.percentile(fv, 90)),
            "p99": float(np.percentile(fv, 99)),
            "p99.9": float(np.percentile(fv, 99.9)),
            "max": float(np.max(fv)),
            "P(>=150k)": self.p_above(150_000),
            "P(>=250k)": self.p_above(250_000),
            "P(>=500k)": self.p_above(500_000),
            "P(<100k)": float(np.mean(fv < 100_000)),
            "best_day_p99": float(np.percentile(self.best_daily_return, 99)),
            "best_week_p99": float(np.percentile(self.best_weekly_return, 99)),
            "resets_mean": float(np.mean(self.n_resets)),
        }


def _t_shocks(rng, shape, df):
    """Standardisierte Student-t-Schocks (Varianz 1)."""
    z = rng.standard_t(df, size=shape)
    return z / np.sqrt(df / (df - 2.0))


def simulate(policy: Policy, market: MarketModel | None = None, n_sims: int = 20_000,
             n_days: int | None = None, steps_per_day: int = 8,
             seed: int = 7) -> SimResult:
    market = market or MarketModel()
    n_days = n_days or len(rules.trading_days())
    rng = np.random.default_rng(seed)

    N = n_sims
    ns = policy.stock.n_positions
    nl = policy.leverage.n_products
    dt = 1.0 / (TRADING_DAYS_PER_YEAR * steps_per_day)

    cash = np.full(N, rules.START_CAPITAL)
    stock_val = np.zeros((N, max(ns, 1)))
    t_S = np.ones((N, max(nl, 1)))          # Basiswert des Turbos, normiert
    t_K = np.zeros((N, max(nl, 1)))         # Knock-out-Barriere
    t_mult = np.zeros((N, max(nl, 1)))      # Stueck * Ratio
    t_alive = np.zeros((N, max(nl, 1)), dtype=bool)

    values = np.zeros((N, n_days + 1))
    values[:, 0] = rules.START_CAPITAL
    n_resets = np.zeros(N, dtype=int)
    best_day = np.full(N, -np.inf)
    best_week = np.full(N, -np.inf)
    week_start_value = np.full(N, rules.START_CAPITAL)

    def depot_value():
        return cash + stock_val.sum(axis=1) + np.where(t_alive, t_mult * np.maximum(t_S - t_K, 0.0), 0.0).sum(axis=1)

    def deploy(mask):
        nonlocal cash, stock_val, t_S, t_K, t_mult, t_alive
        if not mask.any():
            return
        V = depot_value()

        if nl > 0:
            lev_mv = np.where(t_alive, t_mult * np.maximum(t_S - t_K, 0.0), 0.0).sum(axis=1)
            used = lev_mv if policy.leverage_check == "market_value" else np.zeros(N)
            budget = np.maximum(np.minimum(rules.MAX_LEVERAGE_WEIGHT * V, rules.MAX_LEVERAGE_EUR) - used, 0.0)
            budget = np.minimum(budget, np.maximum(cash - nl * rules.FEE_DERIVATIVE, 0.0))
            free_slots = np.maximum((~t_alive).sum(axis=1), 1)
            per_product = budget / free_slots
            price = policy.leverage.target_price
            units = np.minimum(per_product / price, float(rules.MAX_LEVERAGE_UNITS))
            spend = units * price

            L = policy.leverage.initial_leverage
            d = policy.leverage.direction
            S0 = 1.0
            K = S0 * (1.0 - d / L)          # Long: K < S0, Short: K > S0
            ratio = price / abs(S0 - K)
            open_slot = mask[:, None] & (~t_alive) & (spend[:, None] > 1.0)
            if open_slot.any():
                t_S = np.where(open_slot, S0, t_S)
                t_K = np.where(open_slot, K, t_K)
                # gezahlter Spread ist sofort verloren -> weniger Multiplikator
                # Gap-Risiko einpreisen: bei fetten Raendern zahlt ein Turbo
                # bei Barrierenbruch 0 statt negativ. Ohne Korrektur schenkt
                # das Modell hohen Hebeln eine Rendite, die es real nicht gibt
                # (x130 ueber eine Woche: rund +22 %). Der Emittent holt sich
                # das ueber Finanzierungsspread und Stop-Loss-Schwelle zurueck.
                fair = 1.0
                if policy.leverage.fair_pricing:
                    from .optimizer import fair_value_factor
                    fair = fair_value_factor(L, policy.leverage.underlying_annual_vol,
                                             5 if policy.leverage.redeploy_weekly else 20, d)
                t_mult = np.where(open_slot,
                                  (units[:, None] * ratio) * (1.0 - policy.leverage.spread_pct)
                                  / max(fair, 1e-6),
                                  t_mult)
                t_alive = t_alive | open_slot
                cash = cash - np.where(open_slot, spend[:, None] + rules.FEE_DERIVATIVE, 0.0).sum(axis=1)

        if ns > 0:
            V = depot_value()
            target = policy.stock.weight_each * V
            empty = mask[:, None] & (stock_val <= 1.0)
            if empty.any():
                want = np.minimum(target[:, None] * np.ones_like(stock_val), np.inf)
                budget_left = np.maximum(cash - ns * rules.FEE_STOCK, 0.0)
                # Reihenfolge: Positionen nacheinander befuellen, Cash respektieren
                for j in range(ns):
                    take = empty[:, j] & (budget_left > 100.0)
                    amt = np.where(take, np.minimum(want[:, j], budget_left), 0.0)
                    stock_val[:, j] += amt
                    cash -= amt + np.where(take, rules.FEE_STOCK, 0.0)
                    budget_left = np.maximum(budget_left - amt - np.where(take, rules.FEE_STOCK, 0.0), 0.0)

    all_true = np.ones(N, dtype=bool)
    deploy(all_true)

    for day in range(n_days):
        # ---- Wochenstart: Ruecksetzer pruefen und Kapital neu einsetzen ----
        if day > 0 and day % 5 == 0:
            V = depot_value()
            wk_ret = V / np.maximum(week_start_value, 1e-9) - 1.0
            best_week = np.maximum(best_week, wk_ret)

            if policy.use_resets:
                do_reset = V < policy.reset_below
                if do_reset.any():
                    cash = np.where(do_reset, rules.START_CAPITAL, cash)
                    stock_val[do_reset] = 0.0
                    t_alive[do_reset] = False
                    t_mult[do_reset] = 0.0
                    n_resets += do_reset.astype(int)
            week_start_value = depot_value()
            if policy.leverage.redeploy_weekly:
                deploy(all_true)
            else:
                deploy(np.zeros(N, dtype=bool))

        v_open = depot_value()

        # ---- Intraday-Schritte (Barrierenpruefung fuer Knock-outs) ----
        for _ in range(steps_per_day):
            # Schock und Drift getrennt halten: der Hebel-Sleeve darf die
            # Marktdrift nicht mitverstaerken (siehe LeverageSleeve).
            zm = _t_shocks(rng, (N,), market.t_df)
            shock_m = market.annual_vol * np.sqrt(dt) * zm
            crash = rng.random(N) < market.crash_prob_daily / steps_per_day
            shock_m = shock_m + crash * (market.crash_mean + market.crash_vol * rng.standard_normal(N))
            r_m = market.annual_drift * dt + shock_m

            if ns > 0:
                zi = _t_shocks(rng, (N, ns), 3.5)
                r_s = (policy.stock.beta * r_m[:, None]
                       + policy.stock.annual_vol * np.sqrt(dt) * zi)
                jump = rng.random((N, ns)) < policy.stock.jump_prob / steps_per_day
                up = rng.random((N, ns)) < policy.stock.effective_up_share
                r_s = r_s + jump * np.where(up, policy.stock.jump_up, policy.stock.jump_down)
                stock_val *= np.maximum(1.0 + r_s, 0.0)

            if nl > 0:
                zu = _t_shocks(rng, (N, nl), market.t_df)
                r_u = (policy.leverage.underlying_drift * dt
                       + policy.leverage.underlying_beta * shock_m[:, None]
                       + policy.leverage.underlying_annual_vol * np.sqrt(dt) * zu)
                t_S = t_S * (1.0 + r_u)
                d = policy.leverage.direction
                knocked = t_alive & ((t_S <= t_K) if d > 0 else (t_S >= t_K))
                if knocked.any():
                    t_alive = t_alive & ~knocked
                    t_mult = np.where(knocked, 0.0, t_mult)

        v_close = depot_value()
        values[:, day + 1] = v_close
        best_day = np.maximum(best_day, v_close / np.maximum(v_open, 1e-9) - 1.0)

    V = depot_value()
    best_week = np.maximum(best_week, V / np.maximum(week_start_value, 1e-9) - 1.0)
    return SimResult(V, best_day, best_week, n_resets, values)


def simulate_two_depots(policy_a: Policy, policy_b: Policy,
                        market: MarketModel | None = None, n_sims: int = 20_000,
                        seed: int = 7) -> dict:
    """Beide Depots zaehlen in der Rangliste -- fuer den Gesamtsieg zaehlt das bessere."""
    a = simulate(policy_a, market, n_sims, seed=seed)
    b = simulate(policy_b, market, n_sims, seed=seed + 1)
    best = np.maximum(a.final_values, b.final_values)
    return {
        "depot_a": a, "depot_b": b,
        "best_final": best,
        "P(best>=250k)": float(np.mean(best >= 250_000)),
        "P(best>=500k)": float(np.mean(best >= 500_000)),
        "best_day_overall": np.maximum(a.best_daily_return, b.best_daily_return),
        "best_week_overall": np.maximum(a.best_weekly_return, b.best_weekly_return),
    }
