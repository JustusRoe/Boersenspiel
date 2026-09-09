"""Wie hoch kann die Tagesperformance ueberhaupt werden?

Anlass: ich hatte frueher in dieser Sitzung behauptet, fuer den Tagessieg
brauche es rund +300 bis +500 %. Diese Zahl stammte aus einer Rechnung ohne
die 20-%-Grenze fuer Hebelprodukte -- und mit ihr ist sie unerreichbar.

Die Schranke ist einfach und hart:

    Tagesgewinn = Positionswert zu Tagesbeginn x Rendite des Scheins

Beim Kauf ist der Positionswert auf min(20 % x Depot, 20.000 EUR) gedeckelt.
Ein frisch zurueckgesetztes Depot kann also am naechsten Tag hoechstens
20 % x (Scheinrendite) zulegen. Damit die Tagesperformance +300 % erreicht,
muesste der Schein +1500 % machen -- an einem Tag, ohne die Barriere zu
beruehren.

Der Deckel loest sich erst, wenn die Position ueber mehrere Tage waechst:
die 20-%-Grenze gilt beim Kauf, nicht fuer die Haltedauer.
"""
import sys
sys.path.insert(0, ".")
from strategy.rules import leverage_budget

def scheinrendite(barriere, spot, kaufpreis, spot_neu, steigung=0.93, praemie=0.33):
    """Rendite eines Quanto-Turbos, Preis = 0,93*(Barriere-Kurs) + 0,33 EUR."""
    if spot_neu >= barriere:
        return -1.0
    return (steigung * (barriere - spot_neu) + praemie) / kaufpreis - 1.0

def tagesdeckel(depot, anteil, scheinrendite_):
    return anteil * scheinrendite_

print(__doc__)
DEPOT = 93_500.0
TOPF = leverage_budget(DEPOT)
print(f"Depot {DEPOT:,.0f} EUR  ->  Hebeltopf {TOPF:,.0f} EUR = {TOPF/DEPOT*100:.0f} % des Depots\n")

BARR, SPOT, PREIS = 99.69, 98.98, 1.06     # FC8QQS, Brent
print(f"Beispiel Brent-Short, Barriere {BARR}, Kurs {SPOT}, Kauf zu {PREIS:.2f} EUR")
print(f"{'Brent-Tagesmove':>16}{'Scheinrendite':>15}{'Depot-Tag':>12}")
for move in (-0.02,-0.03,-0.05,-0.08,-0.12,-0.16,-0.20):
    r = scheinrendite(BARR, SPOT, PREIS, SPOT*(1+move))
    print(f"{move*100:15.0f}%{r*100:14.0f}%{tagesdeckel(DEPOT, TOPF/DEPOT, r)*100:11.0f}%")

print(f"\nUmgekehrt: welcher Brent-Move waere fuer welches Tagesziel noetig?")
for ziel in (0.25, 0.50, 1.00, 2.00, 3.00):
    noetig_r = ziel / (TOPF/DEPOT)
    # Preis*(1+r) = 0,93*(B - S) + 0,33  ->  S
    s = BARR - (PREIS*(1+noetig_r) - 0.33)/0.93
    print(f"  Depot {ziel*100:+5.0f} %  ->  Schein {noetig_r*100:+6.0f} %  ->  Brent auf "
          f"{s:7.2f} USD ({(s/SPOT-1)*100:+6.1f} %)")

print("\nMit gewachsener Position (Grenze gilt nur beim Kauf):")
for anteil in (0.20, 0.35, 0.50, 0.65):
    for ziel in (1.00, 3.00):
        r = ziel/anteil
        s = BARR - (PREIS*(1+r) - 0.33)/0.93
        print(f"  Positionsanteil {anteil*100:3.0f} %, Ziel {ziel*100:+4.0f} %  ->  Schein {r*100:+6.0f} %"
              f"  ->  Brent {(s/SPOT-1)*100:+6.1f} %")
