# Bugfix Archive (Kurz)

Dieses Archiv fasst die alten Fix-/Entscheidungsnotizen kurz zusammen.

## Wesentliche Fixes
- **SVO Folder Selection Fix**: Initialisierungsreihenfolge und Signal-Timing korrigiert, damit Folder-Wechsel sauber laedt.
- **BBox/Active-Contour Fixes**: Doppelte BBox-Zeichnung entfernt, ROI-Fluss bereinigt, Snake-Initialisierung verbessert.
- **Parameter-Management Umbau**: Algorithmus-spezifische Slider-Aktivierung, Reset-Logik, auto-updated Werte.

## Architekturentscheidung
- **Separate SVO GUI statt tiefer Integration**: geringere Komplexitaet, schneller stabil, klarere Wartung.

## Eher unwichtige bzw. historisch-technische Reports
- Mehrere Zwischenreports zu einzelnen UI-Tunings und Iterationen sind fuer den Tagesbetrieb nicht kritisch.
- Tiefe Detailplaene (z. B. fruehe Depth-Visualisierungsentwuerfe) wurden als Verlauf dokumentiert, nicht als Pflicht-Referenz.

## Hinweis
Die detailreichen Original-Notizen wurden nach dieser Konsolidierung entfernt, um Redundanz und Pflegeaufwand zu reduzieren.
