# Trader 2026 — Strategie

Werkzeuge und Analyse für das Börsenspiel **Trader 2026** der Société Générale
(07.09.2026 – 30.10.2026, 40 Handelstage, 8 Spielwochen, 2 Depots à 100.000 €).

Alle Zahlen unten stammen aus `data/study_report.txt`, erzeugt von
`scripts/run_study.py` mit 40.000 Monte-Carlo-Pfaden je Variante. Die Läufe
sind reproduzierbar — alle Seeds sind deterministisch.

---

## 1. Die Zielfunktion — und warum sie alles umdreht

Gesamtsieger wird, wer am letzten Spieltag **den absolut höchsten Depotwert**
hat. Es gibt keinen Preis für Platz 300 und keinen für einen guten Sharpe.
Bei mehreren tausend Teilnehmern liegt die Siegschwelle erfahrungsgemäß bei
**+150 % bis +400 %**.

Zu maximieren ist damit nicht der Erwartungswert, sondern **P(Top 0,01 %)** —
also die rechte Verteilungsschulter. Eine Strategie mit höherem Median und
niedrigerem 99,9-%-Quantil ist in diesem Spiel *strikt schlechter*.

Was die Simulation dazu sagt (je Depot, über das volle Spiel):

| Strategie | Median | p99,9 | P(≥250k) | P(<100k) |
|---|---:|---:|---:|---:|
| Buy & Hold Index, kein Hebel, kein Rücksetzer | 99.312 | 126.707 | **0,00 %** | 53,4 % |
| Solide Aktien + milder Hebel, mit Rücksetzer | 106.928 | 161.826 | **0,00 %** | 23,1 % |
| Hochvola-Aktien + scharfer Hebel, **ohne** Rücksetzer | 89.990 | 298.049 | 0,38 % | 60,2 % |
| Hochvola-Aktien + scharfer Hebel, **mit** Rücksetzer | 118.454 | 301.759 | 0,47 % | 22,9 % |
| **Maximale Konvexität (x60) mit Rücksetzer** | **126.156** | **392.029** | **3,09 %** | 23,9 % |

Ein solides, vernünftig diversifiziertes Depot hat eine Siegwahrscheinlichkeit
von **exakt null**. Das ist der wichtigste Satz des ganzen Dokuments.

---

## 2. Vier strukturelle Hebel im Regelwerk

### 2.1 Der Rücksetzer ist eine kostenlose Put-Option

> „pro Woche jedes Depot einmal zurücksetzen" (§3)

8 Wochen × 2 Depots = bis zu **16 kostenlose Neustarts**. Dein Downside ist
damit nicht −100 %, sondern **0 %**. Formal wird das Endergebnis zu einem
`max()` über alle Pfade statt zum Erwartungswert eines Pfades.

| Rücksetzer-Regel | Median | P(≥250k) | P(<100k) |
|---|---:|---:|---:|
| gar nicht zurücksetzen | 85.793 | 1,64 % | 63,0 % |
| **Reset bei < 100.000 €** | **126.500** | **3,43 %** | **23,7 %** |
| Reset erst bei < 80.000 € | 113.492 | 2,38 % | 34,5 % |
| Reset schon bei < 130.000 € | 112.010 | 3,40 % | 38,7 % |

Der rationale Schwellenwert ist **exakt 100.000 €** — genau der Punkt, ab dem
der Rücksetzer geschenktes Kapital statt Verlust ist. Früher zurückzusetzen
bringt bei P(Sieg) nichts mehr und wirft nur Substanz weg.

**Die Konsequenz ist unbequem, aber zwingend:** Weil der Verlust bei 0 %
gekappt ist, ist maximales Risiko nicht nur vertretbar, sondern *dominant*.
In Studie 3 steigt mit dem Hebel nicht nur die rechte Schulter, sondern auch
der **Median** — das ist kein Rechenfehler, sondern die Untergrenze: unter
einer Untergrenze verbessert mehr Varianz jedes Quantil oberhalb davon.

### 2.2 Die 20.000-€-Kappe ist knapp und nicht regenerierbar

Für gehebelte Derivate gilt 20 % des Depots **und** absolut 20.000 €, jeweils
**zum Kaufzeitpunkt**. Drei nicht offensichtliche Folgen:

1. **Verkaufe nie einen Gewinner-Turbo.** Wächst er von 20k auf 120k, ist das
   regelkonform — die Prüfung war beim Kauf. Verkaufst du ihn, kannst du diese
   Exposure *nie wieder* aufbauen: Neukäufe bleiben für immer bei 20.000 €,
   egal ob das Depot 100k oder 500k wert ist.
2. **Die Kappe schrumpft relativ.** Bei 100k Depot sind 20.000 € = 20 %
   Wirkung, bei 300k Depot nur noch 6,7 %.
3. **Der Preis je Schein ändert die Konvexität pro Euro *nicht*** — das
   Bezugsverhältnis gleicht das aus. Die 20.000-Stück-Regel kostet nur dann
   etwas, wenn sie verhindert, die vollen 20.000 € einzusetzen:

   | Verfügbare Scheine | Kappe je Wertpapier | Kappe je Depot |
   |---|---:|---:|
   | ab 1,00 € | 20.000 € investiert, 3,75 % | identisch, 3,75 % |
   | nur 0,10–0,25 € | 20.000 € investiert, 2,37 % | **nur 3.333 € investierbar, ~0 %** |

   **Handlungsregel:** Kauf Scheine ab 1,00 €, dann ist die Auslegungsfrage
   irrelevant und du bist gegen beide Lesarten immun. Wer Billigscheine jagt,
   muss sie am ersten Spieltag klären (siehe §6).

### 2.3 Die anderen 80 % sind der eigentliche Renditemotor

Die Hebelkappe gilt nur für *gehebelte* Derivate. Für Aktien gilt nur die
20-%-Grenze je Titel — also 4–5 Positionen ohne jede Volatilitätsgrenze. Ein
Titel mit +100 % an einem Tag bringt +20 % Depotperformance.

Der Screener vom 06.09.2026 findet in Stuttgart handelbare Titel mit **70–144 %
annualisierter Vola** (AXTI 134, AEHR 126, NBIS 144, MSTR 106). Das ist die
Kalibrierungsgrundlage der Simulation.

### 2.4 Zwei Depots sind zwei Lose, kein Diversifikationsargument

| Aufstellung | P(bestes ≥ 250k) | bester Tag p99 | beste Woche p99 |
|---|---:|---:|---:|
| beide identisch, moderat | 0,95 % | 40,7 % | 61,8 % |
| Compounder + Sniper | 3,76 % | 51,7 % | 89,0 % |
| **beide maximal aggressiv** | **6,42 %** | **57,0 %** | **98,7 %** |

Für den Gesamtsieg zählt nur das *bessere* Depot. Diversifikation zwischen den
Depots senkt die Siegchance, sie erhöht sie nicht.

---

## 3. Turbo oder Faktor-Zertifikat für die 20.000 €?

20.000 € Einsatz, DAX-Vola 16 %, eine Position über das volle Spiel gehalten:

| Produkt | Median | p99 | p99,9 | P(Totalverlust) |
|---|---:|---:|---:|---:|
| Faktor-Zertifikat 5x | 19.057 | 39.837 | 51.247 | 0,0 % |
| Faktor-Zertifikat 10x | 16.418 | 71.474 | 117.676 | 0,0 % |
| **Faktor-Zertifikat 15x** | 12.767 | 116.792 | **237.891** | **0,2 %** |
| KO-Turbo x25 | 0 | 94.966 | 123.697 | 50,3 % |
| **KO-Turbo x70** | 0 | 195.864 | **269.662** | **79,6 %** |

Der scharfe Turbo behält auch über 40 Tage die größte rechte Schulter. Er
erkauft sie aber mit 79,6 % statt 0,2 % Totalverlust — das Faktor-Zertifikat
holt rund **88 % des Extremszenarios bei praktisch keinem Ausfall**, und vor
allem: es blockiert den Hebel-Slot nicht.

**Konsequenz:** Faktor-Zertifikat als dauerhafte Hebelbasis im Compounder,
scharfe Turbos als terminierte Ereigniswette im Sniper.

---

## 4. Die Strategie

**Depot A — „Compounder"** (zielt auf den Gesamtsieg)
- 20.000 € in ein **Faktor-Zertifikat 10–15x**, Richtung nach Marktlage.
  Kein Knock-out ⇒ der Slot überlebt acht Wochen und zinst auf.
- 80.000 € auf **4 Aktien à 20 %** aus dem Screener (höchste Konvexitätsscores).
- **Gewinner nie trimmen.** Verlierer rotieren, Gewinner laufen lassen.
- Rücksetzer nur, wenn der Montagswert unter 100.000 € liegt.

**Depot B — „Sniper"** (zielt auf Wochensieg + beste Tagesperformance)
- Jeden Montag früh zurücksetzen, sofern unter 100.000 €.
- 20.000 € in **KO-Turbos mit Hebel 40–80**, gezielt auf einen Katalysatortermin.
- 80.000 € auf 4 Event-Aktien (Zahlen, FDA-Termine, Squeeze-Kandidaten).
- Explodiert B, tauschen die Rollen: B wird zum Compounder, A zum Sniper.

**Für die beste Tagesperformance** muss das Depot *klein* sein: 20.000 €
Hebelbudget sind bei 100k Depot 20 % Wirkung, bei 300k nur 6,7 %. Der Preis
ist strukturell an ein frisch zurückgesetztes 100k-Depot an einem
Ereignistag gebunden.

### Wochen-Playbook

| Wann | Was |
|---|---|
| **Montag 08:00** | Rücksetzer-Entscheidung beide Depots (`daily_briefing.py`). Kapital neu einsetzen. |
| **Di–Do** | Nur nachziehen: Verlierer raus, Gewinner unangetastet. Kaufbudget für Katalysatortage sparen. |
| **Katalysatortag** | Sniper-Depot auf den Termin ausrichten. Turbo + sofort Verkaufslimit-Leiter. |
| **Freitag** | Nichts absichern. Der Wochenwert wird Sonntagnacht gezogen — Absicherung kostet nur Konvexität. |

### Tages-Playbook (deine 3–5 Checkpoints)

1. **08:00** — Briefing lesen, Übernacht-Gaps prüfen, Rücksetzer-Entscheidung.
2. **12:00** — Europäische Bewegung, Limit-Orders nachziehen.
3. **15:30** — US-Eröffnung, der wichtigste Slot. Screener-Kandidaten handeln.
4. **20:00** — bei FOMC/EZB-Terminen: der Entscheid liegt vor 22:00, also handelbar.
5. **21:45** — Schlusskontrolle. Der 22:00-Wert geht in die Tageswertung.

**Wichtiger Mechanik-Trick:** Die 5-Minuten-Haltefrist blockiert die
*Ausführung*, nicht die *Platzierung*. Du kannst im Kaufmoment sofort eine
Take-Profit-Leiter hinterlegen (+100 %, +300 %) — unverzichtbar, weil du nicht
14 Stunden am Bildschirm sitzt und Bots verboten sind.

---

## 5. Weitere legale Mikro-Edges

- **Handelszeit 8:00–22:00** deckt die komplette US-Session ab. US-Nebenwerte
  sind in Stuttgart bis 22:00 handelbar.
- **Limit-Fishing:** 20 Kauforders/Tag/Depot, 7 Tage gültig, Verkäufe
  unbegrenzt. Weit unter Markt liegende Limits auf illiquide Stuttgarter Werte
  kosten nichts und fangen Flash-Dips. (Zu prüfen: ob das Spiel Cash für offene
  Limits blockiert.)
- **Asymmetrie Kauf/Verkauf:** Käufe sind auf 20/Tag limitiert, Verkäufe nicht.
  Breit einsteigen, aggressiv rausrotieren.
- **Gebühren sind irrelevant:** 10 € je Aktien-, 3,90 € je Derivate-Ausführung
  gegen 100.000 € Depot. Kein Grund, Trades zu vermeiden.

---

## 6. Testplan für den ersten Spieltag

Drei Regelambiguitäten lassen sich mit Kleinstorders klären. Was die
Ordermaske zulässt, ist erlaubt — die Regelprüfung des Spiels ist die
maßgebliche Instanz.

1. **Gilt die 20.000-Stück-Kappe je Wertpapier oder je Depot?**
   Kauf 20.000 Stück eines Scheins zu ~0,10 €, dann nochmal 20.000 Stück eines
   *anderen*. Geht der zweite durch, gilt sie je Wertpapier.
2. **Wird die 20-%-Hebelquote auf Marktwert oder Anschaffungskosten geprüft?**
   Nach einem deutlichen Kursgewinn im Hebel-Sleeve einen weiteren Kleinkauf
   versuchen. Geht er durch, zählt die Kostenbasis — dann wächst das
   Kaufbudget mit dem Depot.
3. **Welche Produkte flaggt das System als „gehebelt"?**
   Je eine Kleinorder auf Faktor-Zertifikat, Mini-Future, Open-End-Turbo,
   Inline-Optionsschein und gehebelten ETC. Findet sich ein effektiv
   gehebeltes Produkt, das *nicht* in den 20.000-€-Topf zählt, ist das die
   größte legale Lücke im Regelwerk.

---

## 6a. SG-Produktdaten exportieren — Anleitung

Der Produktfinder auf sg-zertifikate.de deckelt **jeden Export bei 5.000
Produkten**, das Gesamtuniversum umfasst rund 300.000. Ein ungefilterter
Export wird alphabetisch abgeschnitten und enthält dann nur A-Basiswerte
(1&1 bis AMD) — also kein einziges Indexprodukt.

### Spalten, die zwingend aktiviert sein müssen

Der Standardexport liefert nur WKN, Basiswert, Produktart, Bewertungstag,
Kurs Basiswert, Geld, Brief. Damit ist **kein einziges Produkt bewertbar**.
Zusätzlich nötig:

| Spalte | Wofür |
|---|---|
| **Basispreis / Knock-out-Schwelle** | Der Abstand zur Barriere *ist* der Hebel. Ohne sie geht nichts. |
| **Long/Short bzw. Call/Put** | Nicht ableitbar: der Preis ist \|Spot − KO\| in beide Richtungen. |
| **Bezugsverhältnis** | Sonst ist der Hebel um Faktor 10 mehrdeutig. |
| **Hebel / Omega** | Kreuzprobe gegen den abgeleiteten Wert. |
| Laufzeit / Bewertungstag | Restzeitwert bei Standard-Optionsscheinen. |

### Produktarten

| Produktart | Nehmen? | Begründung |
|---|---|---|
| BEST Turbo-Optionsscheine (Open-End) | **ja** | Höchster Hebel, klare Barriere, kein Zeitwertverlust |
| Unlimited Turbo-Optionsscheine (Mini) | **ja** | Wie oben, mit Stop-Loss-Schwelle |
| Faktor-Optionsscheine | **ja** | Kein Knock-out → der Hebel-Slot überlebt acht Wochen (§3) |
| Standard-Optionsscheine (Call/Put) | optional | Konvex, aber Zeitwertverlust; nur für Termine ≤ 2 Wochen |
| Inline-Optionsscheine | Test | Auszahlung auf 10 € gedeckelt, ein 0,50-€-Schein wäre dennoch 20× |
| Discount-, Bonus-, Express-Zertifikate | nein | Gedeckelte Upside — das Gegenteil der Zielfunktion |
| **Classic Aktienanleihen** | **nein** | Nach §4 der Spielregeln nicht handelbar (Stückzinsen) |

### Empfohlene Exportscheiben

Jede bleibt unter dem 5.000er-Limit:

1. **DAX** — nur BEST + Unlimited Turbos (FOMC/EZB-Wetten)
2. **DAX, Nasdaq 100, S&P 500** — nur Faktor-Optionsscheine (Compounder-Basis)
3. **S&P 500, Nasdaq 100** — nur Turbos
4. **Einzelaktien** aus `data/screen_latest.csv` — Turbos + Faktor

Alle Dateien nach `data/sg/` legen, dann:

```python
from strategy.optimizer import load_products_dir, optimise_basket

produkte, diag = load_products_dir("data/sg")
korb = optimise_basket(produkte, depot_value=100_000, horizon_days=5,
                       vol_by_underlying={"DAX": 0.16, "S&P 500": 0.14},
                       objective="p_target", target_multiple=1.5)
print(korb.to_table())
```

Der Importer verwirft Produkte, die er nicht sauber bewerten kann, statt
Werte zu raten, und nennt für jede verworfene Zeile den Grund.

### Offene Frage an den Produktfinder

Gibt es eine eigene Kategorie **„Faktor-Zertifikate"** (nicht
„Faktor-*Optionsscheine*")? Die Spielregeln zählen als gehebelt nur
„Optionsscheine, Turbo-Optionsscheine und Faktor-Optionsscheine" auf. Ein
gehebeltes Produkt außerhalb dieser Namen fiele nicht unter die
20.000-€-Kappe — das wäre die größte legale Lücke im Regelwerk.

---

## 7. Was ausdrücklich verboten ist

Ziffer 10 der Spielregeln, ohne Interpretationsspielraum:

- **Keine automatisierten Handelssysteme.** Dieses Repo erzeugt *Signale und
  Vorschläge*; jede Order gibst du von Hand auf.
- **Kein zweiter Benutzername.** Führt zum sofortigen Ausschluss beider Depots.
- **Keine Trades, deren Zweck die Verkleinerung des Depotwerts ist**, um die
  20-%-Quote zu hebeln. Das ist namentlich als Manipulation gelistet.

Die Strategie oben braucht nichts davon.

---

## 8. Marktlage bei Spielstart (06.09.2026)

| Markt | Stand | 20T | realisierte Vola 20T |
|---|---:|---:|---:|
| DAX | 26.046 | −1,0 % | 8,9 % |
| S&P 500 | 7.719 | −0,5 % | 8,3 % |
| Nasdaq 100 | 29.544 | −0,6 % | 12,9 % |
| VIX | 14,53 | — | Jahrestief 2026 |
| Brent | 96,28 | **+15,2 %** | 34,1 % |
| US 10Y | 4,78 % | +2,7 % | — |

**These:** Konvexität ist gerade historisch billig. Niedrige implizite Vola
heißt mehr Hebel pro Euro. Gleichzeitig gab es in *jedem* Midterm-Wahljahr seit
1990 einen Rücksetzer von ≥7 % zwischen Mitte August und Mitte Oktober — das
liegt exakt im Spielfenster. Long Vola statt Long Momentum.

**Katalysatoren:** EZB 10.09. · **FOMC 16.09.** (20:00, voll handelbar) ·
Triple Witching 18.09. · Quartalsende 30.09. · Q3-Berichtssaison ab 13.10. ·
**FOMC 28.10.** (zwei Tage vor Spielende) · EZB 29.10.

---

## 9. Repo benutzen

```bash
pip install -r requirements.txt

# Tages-Briefing: Marktlage, Katalysatoren, Rücksetzer-Empfehlung, Regelbudgets
python3 scripts/daily_briefing.py --depot-a 100000 --depot-b 100000

# Vollständige Strategiestudie neu rechnen (~15 min bei 40.000 Pfaden)
python3 scripts/run_study.py --sims 40000
python3 scripts/run_study.py --quick        # schnelle Variante
```

Hebelkorb aus einem echten SG-Produktfinder-Export optimieren:

```python
from strategy.optimizer import load_products_csv, optimise_basket

produkte = load_products_csv("data/sg_export.csv")
korb = optimise_basket(produkte, depot_value=100_000, horizon_days=5,
                       vol_by_underlying={"DAX": 0.16},
                       objective="p_target", target_multiple=1.5)
print(korb.to_table())
```

Der CSV-Import mappt Spaltennamen tolerant und versteht deutsche
Dezimalkommas. Faktor-Zertifikate werden am Typ erkannt und pfadabhängig
bewertet; Turbos ohne Knock-out-Angabe werden übersprungen statt geraten.

### Module

| Modul | Zweck |
|---|---|
| `strategy/rules.py` | Spielregeln als ausführbare Nebenbedingungen |
| `strategy/simulator.py` | Monte-Carlo des Spiels inkl. Rücksetzer-Option |
| `strategy/optimizer.py` | Konvexitätsoptimierung unter den drei Kappen |
| `strategy/data.py` | Marktdaten (Yahoo) über `requests` |
| `strategy/screener.py` | Kandidatensuche für den Aktien-Sleeve |
| `strategy/calendar_events.py` | Katalysatorkalender des Spielzeitraums |

---

## 10. Grenzen des Modells — ehrlich

- **Die Simulation unterstellt keinen Selektionsvorteil.** Die Sprungkomponente
  ist standardmäßig erwartungswertneutral; die Turbos sind arbitragefrei
  bepreist (inklusive Gap-Risiko — ohne diese Korrektur schenkt sich ein
  x130-Turbo über eine Woche rund 22 % Gratisrendite, die es real nicht gibt).
  Die Zahlen sagen also, was **die Struktur des Spiels** hergibt, nicht was
  gutes Stock-Picking hergibt. Studie 1, Zeile 6 zeigt, was ein unterstellter
  Vorteil zusätzlich brächte: P(≥250k) von 3,09 % auf 3,96 %.
- **Kostenlose Kursdaten sind gegenüber den EUWAX-Echtzeitkursen des Spiels
  verzögert.** Sie taugen für Screening und Kalibrierung, nicht für
  Sekunden-Timing beim Ordern.
- **Kein automatischer SG-Produktabruf.** Die Produktsuche auf
  sg-zertifikate.de ist eine SPA hinter einer WAF; der Optimizer arbeitet
  deshalb mit CSV-Export. Ein Browser-Scrape scheitert in dieser Umgebung
  zusätzlich am Egress-Proxy.
- **Die Volatilitätsannahmen sind statisch.** Reale Vola clustert; nach einem
  Crash sind Scheine teurer, als das Modell unterstellt.
- **Siegschwellen sind geschätzt.** P(≥250k) ist ein Proxy für die Siegchance,
  keine Aussage über das tatsächliche Teilnehmerfeld.
