# Edge Pipelines - Zusammenfassung

## 1) Adaptive Pipeline (Hauptansatz)
Wesentliche Punkte aus `ADAPTIVE_PIPELINE_README`, `PIPELINE_EXPLAINED`, `INTELLIGENT_PREPROCESSING`:
- Adaptive Vorverarbeitung statt fixer Slider-Entscheidungen
- Bedingte CLAHE- und Bilateral-Nutzung anhand Bildmetriken
- Multi-Threshold-Canny + geometrische/depth-basierte Konturvalidierung
- Temporales Scoring (Exponential Decay) gegen flackernde False Positives
- Debug-Ausgabe pro Stage mit Timing und Entscheidungsgruenden

Typische Einordnung:
- Hoehere Robustheit, etwas mehr Laufzeit
- Besonders sinnvoll bei Rauschen, Low-Contrast und Frame-Sequenzen

## 2) Simple Pipeline (Vergleichsansatz)
Aus `SIMPLE_VS_COMPLEX_PIPELINE`:
- Weniger Schritte, schnellere Laufzeit
- Gute Baseline bei klaren Kanten und geringer Stoerung
- Weniger robust bei schwierigen Szenen

## 3) Interaktiver Debug-Modus
Aus `INTERACTIVE_DEBUG_MODE`:
- Step-by-step Navigation per Tastatur
- Fokus auf einzelne Pipeline-Stufen in grosser Ansicht
- Grid-View weiterhin moeglich fuer Gesamtueberblick

## 4) Praktische Empfehlung
- Start mit Simple Pipeline fuer schnellen Check
- Wechsel auf Adaptive Pipeline, sobald Stabilitaet/Robustheit wichtiger ist
- Debug-Modus nutzen, wenn Entscheidungen (CLAHE/Bilateral/Filter) nachvollzogen werden muessen
