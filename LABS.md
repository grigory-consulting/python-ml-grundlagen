# Übungen: Python, Datenanalyse und Machine Learning

Zu jedem Teil der Folien gibt es ein Übungs-Notebook. Dieses Dokument zeigt, wie Sie die Notebooks öffnen, wie sie aufgebaut sind und welche Aufgaben Sie in welchem Lab erwarten. Die Aufgaben selbst, alle Tipps und die Kontrollergebnisse stehen im jeweiligen Notebook.

## So öffnen Sie ein Lab

0. Öffnen Sie den Kursordner in VS Code. Die Übungen liegen im Ordner `labs/`, die Daten im Ordner `data/`.
2. Öffnen Sie das Notebook des aktuellen Teils, zum Beispiel `labs/lab_02_numpy_pandas.ipynb`.
3. Wählen Sie oben rechts den Kernel `pyml`. Steht dort `base` oder ein anderer Name, fehlen später Pakete.
4. Führen Sie zuerst die Setup-Zelle ganz oben aus. Sie sucht den Datenordner und legt ihn in der Variablen `DATA` ab. Die Ausgabe lautet `Datenordner: ...`.
5. Führen Sie danach die Zelle mit den Imports aus. Erst dann beginnen die Aufgaben.

Daten laden Sie immer über `DATA`, zum Beispiel `pd.read_csv(DATA / "titanic.csv")`. So findet das Notebook die Dateien, egal aus welchem Ordner es gestartet wurde. Für kein Lab brauchen Sie eine Netzverbindung. Die einzige Ausnahme ist eine gekennzeichnete Zusatzaufgabe in Lab 8.

## So sind die Notebooks aufgebaut

+ **Lernziele:** Am Anfang stehen drei bis fünf Lernziele und der Teil der Folien, zu dem das Lab gehört.
+ **Blöcke:** Jedes Lab besteht aus drei bis sechs Blöcken. Ein Block beginnt mit einer Textzelle, die das Thema nennt und die Aufgaben auflistet. Sie bearbeiten die Blöcke nacheinander zwischen den Folien. Innerhalb eines Blocks steigt die Schwierigkeit: erst nachmachen, dann abwandeln, dann selbst lösen.
+ **Aufgaben mit Kontrollergebnis:** Jede Aufgabe hat eine eigene Code-Zelle. Im Aufgabentext steht unter „Erwartet" ein Kontrollergebnis, zum Beispiel eine Zeilenzahl oder eine Kennzahl. Damit prüfen Sie sich selbst. Stimmt Ihre Zahl nicht, prüfen Sie zuerst, ob Sie `random_state=1` oder den angegebenen Seed gesetzt haben.
+ **Gerüste mit `...`:** Die Code-Zellen enthalten ein Gerüst mit einem Tipp als Kommentar. Den Platzhalter `...` ersetzen Sie durch Ihren Code. Manche Zeilen sind auskommentiert: Entfernen Sie das `#`, sobald die Zeilen darüber fertig sind.
+ **Gerüste laufen auch unverändert:** `...` ist gültiges Python. Eine Zelle, die Sie noch nicht bearbeitet haben, bricht deshalb nicht ab. Sie gibt dann `Ellipsis`, eine leere Tabelle oder gar nichts aus. Sehen Sie `Ellipsis` in einer Ausgabe, ist an dieser Stelle noch ein Platzhalter offen.
+ **Aufholzellen:** Einige Blöcke beginnen mit einer Zelle, die Zwischenergebnisse aus dem vorigen Block noch einmal anlegt. So können Sie weitermachen, auch wenn Sie einen Block nicht abgeschlossen haben.
+ **Zusatzaufgaben:** Am Ende jedes Labs stehen zwei bis vier Zusatzaufgaben für alle, die früher fertig sind. Sie sind freiwillig und haben ebenfalls ein Kontrollergebnis.
+ **Was Sie mitnehmen:** Den Abschluss bilden drei Merksätze zum Lab.

Wenn ein Ergebnis merkwürdig aussieht: Kernel neu starten und alle Zellen von oben nach unten ausführen. Der Kernel merkt sich Variablen in der Reihenfolge der Ausführung, nicht in der Reihenfolge im Notebook.

## Übersicht

| Lab | Datei | gehört zu Teil | Datensätze | Blöcke | Aufgaben |
|---|---|---|---|---|---|
| 0 | `lab_00_setup_check.ipynb` | Einrichtung der Arbeitsumgebung | alle Dateien in `data/` (nur Prüfung), `titanic.csv` (Zusatz) | 3 | 9, dazu 4 Zusatzaufgaben |
| 1 | `lab_01_python_grundlagen.ipynb` | Teil 1: Python-Grundlagen | keine Dateien, Beispielwerte im Notebook | 6 | 23, dazu 3 Zusatzaufgaben |
| 2 | `lab_02_numpy_pandas.ipynb` | Teil 2: NumPy und pandas | `versicherte.csv`, Zufallszahlen | 6 | 23, dazu 3 Zusatzaufgaben |
| 3 | `lab_03_eda_visualisierung.ipynb` | Teil 3: Daten einlesen, aufbereiten, explorieren und visualisieren | `titanic.csv`, `winequality-red.csv`, `versicherte.csv`, `versicherte.db` | 6 | 23, dazu 3 Zusatzaufgaben |
| 4 | `lab_04_erstes_modell.ipynb` | Teil 4: Grundlagen von Machine Learning und KI | Iris (in scikit-learn enthalten), `titanic.csv` | 3 | 11, dazu 3 Zusatzaufgaben |
| 5 | `lab_05_regression_klassifikation.ipynb` | Teil 5: Überwachtes Lernen | erzeugte Punkte, `california_housing.csv`, Brustkrebs-Datensatz (in scikit-learn enthalten) | 5 | 16, dazu 3 Zusatzaufgaben |
| 6 | `lab_06_evaluation_tuning.ipynb` | Teil 6: Modelltraining, Feature Engineering, Hyperparameter und Evaluation | `titanic.csv`, erzeugte Punkte, Brustkrebs-Datensatz, `california_housing.csv`, künstliche Klassifikationsdaten, `versicherte.csv` (Zusatz) | 6 | 18, dazu 3 Zusatzaufgaben |
| 7 | `lab_07_clustering.ipynb` | Teil 7: Unüberwachtes Lernen | erzeugte Punktwolken, `versicherte.csv` | 5 | 17, dazu 3 Zusatzaufgaben |
| 8 | `lab_08_deep_learning.ipynb` | Teil 8: Einführung in Deep Learning | FashionMNIST (liegt in `data/`) | 6 | 18, dazu 4 Zusatzaufgaben |
| 9 | `lab_09_workflow_pipeline.ipynb` | Teil 9: Der ML-Workflow mit scikit-learn | `titanic.csv`, Zufallszahlen (Zusatz) | 6 | 18, dazu 3 Zusatzaufgaben |
| 10 | `lab_10_praxisprojekt.ipynb` | Teil 10: Praxisbeispiele und Interpretation von Modellen | `ai4i2020.csv`, `titanic.csv`, `california_housing.csv`, im Mini-Projekt ein Datensatz nach Wahl | 4 | 10 geführte Aufgaben und 10 Projektschritte, dazu 3 Zusatzaufgaben |

Teil 11 (Transfer in die Praxis) hat kein eigenes Lab. Die Versichertentabelle `versicherte.csv` ist synthetisch erzeugt und enthält keine echten Personen.

## Lab 0: Arbeitsumgebung prüfen

Datei: `lab_00_setup_check.ipynb`, gehört zur Einrichtung der Arbeitsumgebung.

**Lernziele**

+ Sie prüfen, dass Ihr Notebook mit dem richtigen Python und dem richtigen Kernel läuft.
+ Sie importieren alle Kurspakete und lesen deren Versionen ab.
+ Sie kontrollieren, dass der Datenordner mit allen Dateien vorhanden ist.
+ Sie bedienen ein Notebook sicher: Markdown-Zelle, Reihenfolge der Ausführung, Kernel neu starten, Hilfe aufrufen.
+ Sie erzeugen einen ersten Plot als Funktionstest.

**Block 1: Python, Pakete und Daten**

1. Python-Version und Pfad des Interpreters ausgeben und prüfen, ob der Pfad zur Umgebung `pyml` gehört.
2. Die Prüfzelle so ergänzen, dass alle acht Kurspakete importiert und mit Version ausgegeben werden.
3. Die installierten Versionen mit den Mindestversionen aus `requirements.txt` vergleichen.
4. Prüfen, ob im Datenordner alle neun Kursdateien liegen, und ihre Größe ausgeben.

**Block 2: Notebook bedienen**

5. Eine Markdown-Zelle mit Überschrift, fett gesetztem Wort und Liste anlegen und ausführen.
6. Eine Zelle mehrfach ausführen, den Kernel neu starten und beobachten, was mit der Variablen `beitrag` passiert.
7. Die Hilfe zu `round` auf zwei Wegen aufrufen und eine Zahl auf zwei Nachkommastellen runden.

**Block 3: Erster Plot als Funktionstest**

8. Die vorbereitete Zelle ausführen: eine Sinuskurve mit beschrifteten Achsen.
9. Den Plot abwandeln: Kosinus dazuzeichnen, beide Kurven mit `label` versehen, Legende einblenden.

**Zusatzaufgaben**

+ Z1: `requirements.txt` einlesen und alle Zeilen ausgeben, die ein Paket nennen.
+ Z2: `titanic.csv` mit pandas laden und die Form der Tabelle ausgeben.
+ Z3: Die Hilfe zu `math.sqrt` aufrufen und die Wurzel aus 16 berechnen.
+ Ohne Zelle: Im Anaconda Prompt im Kursordner `git status` und `git pull` ausführen.

## Lab 1: Python-Grundlagen

Datei: `lab_01_python_grundlagen.ipynb`, gehört zu Teil 1: Python-Grundlagen.

**Lernziele**

+ Sie legen Werte in Variablen ab, wandeln Typen um, rechnen mit `//` und `%` und formatieren Ausgaben mit f-Strings.
+ Sie steuern Abläufe mit `if`/`elif`/`else`, `for` (mit `range` und `enumerate`) und `while`.
+ Sie arbeiten mit Listen, Slicing, Tupeln, Sets, Dictionaries (`get`, `items`) und Comprehensions.
+ Sie schreiben eigene Funktionen mit Defaultwerten und rufen sie mit Keyword-Argumenten auf.
+ Sie lesen Fehlermeldungen von unten nach oben und fangen Fehler mit `try`/`except` ab.

Alle Namen, Beträge und Postleitzahlen in diesem Lab sind frei gewählte Beispielwerte.

**Block 1: Variablen, Typen, Operatoren und f-Strings**

1. Die Jahre bis 65 berechnen und den Satz mit einem f-String ausgeben.
2. Zwei Textwerte in Zahlen umwandeln, den Jahresbeitrag berechnen und mit zwei Nachkommastellen ausgeben.
3. 100 Tage mit `//` und `%` in Wochen und Resttage zerlegen.

**Block 2: Bedingungen**

4. Eine Punktzahl mit `if`/`elif`/`else` in eine Stufe übersetzen und mit mehreren Werten testen.
5. Eine verschachtelte Altersprüfung flach mit `and` und `elif` schreiben.
6. Einen Namen prüfen: leer, vorhanden, länger als drei Zeichen. Den Wahrheitswert des Strings nutzen.

**Block 3: Schleifen**

7. Mit `for` und `range` Quadratzahlen ausgeben und die Summe der ersten zehn Quadratzahlen bilden.
8. In einer Liste von Arztbesuchen die geraden Werte zählen und mit `enumerate` die Positionen der Werte über 6 ausgeben.
9. Mit einer `while`-Schleife Beträge summieren, bis eine 0 kommt.

**Block 4: Listen, Slicing, Tupel und List Comprehensions**

10. Aus einer Beitragsliste vier Ausschnitte bilden und ein Tupel in drei Variablen entpacken.
11. Alle Namen mit „A" in eine neue Liste übernehmen: erst mit Schleife und `append`, dann als List Comprehension.
12. Die durchschnittliche Länge der Städtenamen einer Liste berechnen.
13. Duplikate aus einer PLZ-Liste entfernen, Reihenfolge behalten, mit `set(...)` vergleichen.

**Block 5: Dictionaries und Sets**

14. Mit einem Dict zählen, wie oft jede Stadt vorkommt: erst mit `in`, dann mit `get`. Ausgabe mit `items()`.
15. Schnittmengen der Schlüssel und der Werte zweier Dicts bilden und ein Dict umdrehen.
16. In einer Liste aus zwölf verschachtelten Dicts alle Personen unter 30 ausgeben und das Set aller Postleitzahlen bilden.
17. Alle Personen einer vorgegebenen Stadt finden, unabhängig von Groß- und Kleinschreibung, mit Meldung bei null Treffern.

**Block 6: Funktionen, Fehlermeldungen und `try`/`except`**

18. Die Berechnung aus Aufgabe 12 als Funktion `durchschnitt_laenge(woerter)` schreiben.
19. Eine Funktion `gesamtbeitrag(bausteine, rabatt=None)` schreiben und auf drei Arten aufrufen.
20. In einem vorgegebenen Code drei Fehler finden: Fehlermeldung von unten nach oben lesen, korrigieren, nächsten Fehler suchen.
21. Zwei Fehler abfangen: eine Division durch 0 und das Öffnen einer fehlenden Datei, mit `else` und `finally`.

**Zusatzaufgaben**

+ Z1: In einer Zahlenliste alle Paare finden, deren Summe 10 ergibt, jedes Paar nur einmal.
+ Z2: Eine Passwortabfrage mit `while` über eine Liste von Versuchen schreiben und die Versuche zählen.
+ Z3: Mit `random.seed(42)` und `random.randint(0, 9)` zehn wiederholbare Zufallsziffern erzeugen.

## Lab 2: NumPy und pandas

Datei: `lab_02_numpy_pandas.ipynb`, gehört zu Teil 2: NumPy und pandas.

**Lernziele**

+ Sie legen NumPy-Arrays an, rechnen vektorisiert und werten sie mit Masken und `axis` aus.
+ Sie lesen eine CSV-Datei als DataFrame ein und wählen Zeilen und Spalten mit `loc` und `iloc` aus.
+ Sie filtern mit Masken, `query` und `isin` und leiten neue Spalten ab (BMI, Altersgruppe, Geburtsjahr).
+ Sie werten Tabellen mit `groupby` und `agg` aus und wissen, wann der Median besser passt als der Mittelwert.
+ Sie verbinden zwei Tabellen mit `merge` und speichern ein Ergebnis als CSV.

In diesem Lab beginnt die Nummerierung der Aufgaben in jedem Block neu bei 1.

**Block 1: Arrays anlegen und rechnen**

1. Ein Array mit den Zahlen 0 bis 99 anlegen, quadrieren und die Summe bilden.
2. Eine Million Temperaturwerte von Celsius nach Fahrenheit umrechnen: die vektorisierte Fassung ergänzen und die Zeiten vergleichen.
3. `np.linspace(0, 10, 100)` anlegen und `shape`, `dtype`, `min` und `max` ausgeben.

**Block 2: Masken, Aggregationen, Zufall**

1. 1000 normalverteilte Körpergrößen erzeugen und den Anteil über 190 cm bestimmen.
2. Eine Zufallsmatrix mit 5 Zeilen und 3 Spalten anlegen und den Mittelwert je Spalte und je Zeile berechnen.
3. Die Würfelsimulation auf drei Würfel und 10 000 Würfe erweitern: Wie oft ist die Summe mindestens 15?

**Block 3: Einlesen und Auswählen**

1. `versicherte.csv` einlesen und mit `shape`, `head()` und `info()` prüfen: Typen und fehlende Werte.
2. Drei Spalten der Zeilen 10 bis 20 auswählen, einmal mit `loc`, einmal mit `iloc`.
3. Mittelwert, Minimum und Maximum von `arztbesuche_jahr` bestimmen.

**Block 4: Filtern und neue Spalten**

1. Alle männlichen Versicherten mit BMI über 30 auswählen: mit Maske und mit `query`.
2. Mit `isin` alle Versicherten aus den Stadtstaaten auswählen und die Raucherinnen und Raucher zählen.
3. Den BMI aus Größe und Gewicht nachrechnen und mit der vorhandenen Spalte vergleichen.
4. Mit `pd.cut` die Spalten `altersgruppe` und `bmi_klasse` anlegen und die Normalgewichtigen nach Alter sortiert zeigen.

**Block 5: Gruppieren**

1. Durchschnittliche Größe und durchschnittliches Gewicht je Altersgruppe berechnen.
2. Mittlere Leistungsausgaben je Bundesland und Altersgruppe als Kreuztabelle mit `unstack()` darstellen.
3. Den Raucheranteil je Geschlecht und je Bundesland bestimmen.
4. Je Bundesland Mittelwert und Median der Leistungsausgaben vergleichen.

**Block 6: Verbinden, Datum, Speichern**

1. Die Region mit `pd.merge` als Left Join anhängen, Zeilenzahl prüfen, Versicherte je Region zählen.
2. Eine zweite Tabelle mit 1000 zufälligen Versichertennummern bauen, mit `indicator=True` anhängen und die Treffer zählen.
3. `geburtsdatum` in ein Datum umwandeln, Geburtsjahr und Geburtsmonat ableiten und je Monat zählen.
4. Eine Auswertung je Bundesland als CSV mit Semikolon und Dezimalkomma speichern und zur Kontrolle wieder einlesen.

**Zusatzaufgaben**

1. Gruppen filtern: In welchen Bundesländern liegen die mittleren Leistungsausgaben über 3300 Euro?
2. Feinere Altersgruppen mit `pd.cut` anlegen und je Gruppe Anzahl und Median der Leistungsausgaben bestimmen.
3. Den Median je Bundesland per `merge` an jede Zeile anhängen und den Anteil der Versicherten über dem Median ihres Bundeslands bestimmen.

## Lab 3: Daten einlesen, aufbereiten, explorieren, visualisieren

Datei: `lab_03_eda_visualisierung.ipynb`, gehört zu Teil 3: Daten einlesen, aufbereiten, explorieren und visualisieren.

**Lernziele**

+ Sie lesen CSV-Dateien mit den passenden Parametern ein und erkennen typische Einlesefehler (Trennzeichen, PLZ mit führender Null, Datum als Text).
+ Sie prüfen einen Datensatz systematisch: Größe, Typen, Kennzahlen, fehlende Werte als Anzahl und als Anteil.
+ Sie behandeln fehlende Werte, Duplikate, Datentypen und Ausreißer nachvollziehbar und ohne `inplace=True`.
+ Sie bauen Diagramme mit matplotlib (`fig, ax`) und seaborn: Histogramm, Countplot, Boxplot, Scatterplot, Heatmap.
+ Sie belegen eine Aussage mit einem eigenen, sauber beschrifteten Diagramm.

In diesem Lab beginnt die Nummerierung der Aufgaben in jedem Block neu bei 1.

**Block 1: Einlesen**

1. `winequality-red.csv` einmal ohne Parameter und einmal mit `sep=";"` einlesen und `shape` und `dtypes` vergleichen.
2. `versicherte.csv` einmal ohne Parameter und einmal mit `parse_dates` und `dtype={"plz": str}` einlesen und die Postleitzahlen prüfen.
3. Eine kleine Tabelle mit Semikolon und Dezimalkomma in das temporäre Verzeichnis schreiben und korrekt wieder einlesen.
4. Mit `pd.read_sql` aus der SQLite-Datenbank `versicherte.db` lesen: Anzahl und mittlere Leistungsausgaben je Bundesland für Versicherte mit mindestens 10 Arztbesuchen.
5. Stolperstein Spaltenname: Die Spalte `alter` heißt wie das SQL-Schlüsselwort `ALTER`. Erst die Fehlermeldung ansehen, dann den Namen in doppelte Anführungszeichen setzen.

**Block 2: Überblick und fehlende Werte**

1. Überblick über Titanic: `shape`, `info()`, `describe()`, fehlende Werte je Spalte als Anzahl und als Anteil.
2. Zählen, wie viele Zeilen nach `dropna()` ohne Parameter und nach `dropna(subset=["Age"])` bleiben.
3. `Age` einmal mit dem Mittelwert, einmal mit dem Median füllen und `describe()` vorher und nachher vergleichen.
4. Die bereinigte Tabelle `feat_df` bauen: Spalten wählen, Flag-Spalten anlegen, `Age` und `Embarked` füllen, `Cabin` entfernen.

**Block 3: Duplikate, Typen, Ausreißer**

1. Titanic und die Versichertendaten auf doppelte Zeilen prüfen, Duplikate entfernen, Eindeutigkeit der Versichertennummer prüfen.
2. `Sex` und `Embarked` in den Typ `category` umwandeln und die Kategorien ausgeben.
3. Mit Boolean-Masken auswählen: alle Männer mit Ticketpreis über 20 und die fünf ältesten Passagiere der ersten Klasse.
4. Für `Fare` die IQR-Grenzen berechnen, markierte Tickets zählen und `Fare` mit `clip` begrenzen.

**Block 4: matplotlib-Grundgerüst**

1. Ein Histogramm von `Fare` mit Titel und deutschen Achsen zeichnen und als PNG speichern.
2. Zwei Histogramme nebeneinander: `Age` der Überlebenden und der nicht Überlebenden.
3. `Age` mit 10 und mit 50 Klassen nebeneinander zeichnen und den Unterschied in einem Satz beschreiben.

**Block 5: seaborn-Grundformen**

1. Histogramm von `Fare` mit `kde=True`.
2. Countplot von `Pclass` mit `hue="Überlebt"`.
3. Zwei Boxplots nebeneinander: `Fare` nach `Pclass` und `Age` nach `Pclass`, dazu ein Satz zur Beobachtung.
4. Scatterplot `Age` gegen `Fare` mit `hue="Überlebt"` und `alpha=0.7`.

**Block 6: Korrelation, Heatmap und Ihr eigenes Diagramm**

1. Die Korrelation aller Zahlenspalten mit `Survived` berechnen.
2. Aus `feat_df` eine Heatmap der Korrelationsmatrix erzeugen.
3. Abschlussaufgabe: ein eigenes Diagramm bauen, das die Aussage „In der dritten Klasse überlebten weniger Passagiere" belegt, und als PNG speichern.

**Zusatzaufgaben**

1. Die IQR-Grenzen für `Age` berechnen und die markierten Passagiere ansehen: Fehler oder plausible Werte?
2. In den bereinigten Versichertendaten die zehn höchsten Leistungsausgaben ansehen: Fehler oder Hochkostenfälle?
3. Die Leistungsausgaben der Versicherten als Histogramm und als Boxplot je Altersgruppe zeichnen.

## Lab 4: Das erste Modell

Datei: `lab_04_erstes_modell.ipynb`, gehört zu Teil 4: Grundlagen von Machine Learning und KI.

**Lernziele**

+ Beispiele den drei Lernarten zuordnen und Regression von Klassifikation unterscheiden
+ Begründen, wann eine feste Regel genügt und wann sich Machine Learning lohnt
+ Merkmale `X` und Zielgröße `y` aus einer Tabelle auswählen
+ Ein erstes Modell mit `fit`, `predict` und `score` trainieren und prüfen
+ Erklären, warum nur zurückgehaltene Testdaten zeigen, ob ein Modell generalisiert

In diesem Lab beginnt die Nummerierung der Aufgaben in jedem Block neu bei 1.

**Block 1: Lernarten zuordnen, ML oder Regel**

1. Sechs Beispiele einer Lernart zuordnen: überwacht, unüberwacht oder bestärkend.
2. Für fünf überwachte Aufgaben entscheiden: Regression oder Klassifikation?
3. Für fünf Situationen entscheiden: ML oder Regel?
4. In `titanic.csv` eine Zielgröße und drei Merkmale wählen und `X_t` und `y_t` bilden.

**Block 2: Das erste Modell auf Iris**

1. Das erste Modell selbst tippen: Split, `KNeighborsClassifier(n_neighbors=5)`, `fit`, `predict`, `score`.
2. `n_neighbors` variieren (1, 3, 15, 50, 120) und den Score auf den Testdaten notieren.
3. `test_size=0.5` setzen und das Ergebnis vergleichen.
4. Die Art einer neuen Blüte mit vorgegebenen Messwerten vorhersagen.

**Block 3: Testdaten zurückhalten, Modell tauschen**

1. Für k-NN mit k = 1 und für einen Entscheidungsbaum den Score auf Trainings- und Testdaten vergleichen, für drei verschiedene Splits.
2. Drei Modelle in dieselbe Schleife einsetzen und eine Tabelle mit Trainings- und Testscore ausgeben.
3. Die ersten zehn Vorhersagen der logistischen Regression mit der Wahrheit vergleichen.

**Zusatzaufgaben**

1. k-NN für `random_state` 0 bis 9 trainieren und kleinsten, größten und mittleren Testscore ausgeben.
2. Mit `np.bincount` die Arten im Testteil zählen und den Split mit `stratify=y` wiederholen.
3. Für eine vorgegebene Blüte die Klassenwahrscheinlichkeiten mit `predict_proba` ausgeben.

## Lab 5: Regression und Klassifikation

Datei: `lab_05_regression_klassifikation.ipynb`, gehört zu Teil 5: Überwachtes Lernen.

**Lernziele**

+ Den Gradientenabstieg für eine Gerade in NumPy nachbauen und die Wirkung der Lernrate beobachten
+ Eine lineare Regression trainieren, ihre Koeffizienten lesen und nach dem Skalieren vergleichen
+ Daten in der richtigen Reihenfolge vorbereiten: erst teilen, dann skalieren
+ Mit `predict_proba` eine eigene Schwelle setzen und die Folgen in der Konfusionsmatrix ablesen
+ Fünf Klassifikationsverfahren in einer Schleife trainieren und in einer Tabelle nebeneinanderstellen

In diesem Lab beginnt die Nummerierung der Aufgaben in jedem Block neu bei 1.

**Block 1: Gradientenabstieg von Hand**

1. Die Schleife von der Folie übernehmen: Start bei `m = 0`, `b = 0`, Lernrate 0.01, 1000 Schritte.
2. Zählen, nach wie vielen Schritten der MSE unter 1.1 fällt, für zwei Lernraten. Danach die Lernrate 0.05 ausprobieren.
3. Den MSE je Schritt sammeln, die Kurve zeichnen und mit `LinearRegression` vergleichen.

**Block 2: Lineare Regression auf California Housing**

1. Daten teilen, `LinearRegression` trainieren, MSE und R² auf den Testdaten ausgeben.
2. Die Koeffizienten als sortierte `Series` ausgeben und den stärksten positiven und negativen benennen.
3. Den Hauswert für einen vorgegebenen Bezirk vorhersagen, danach `MedInc` um 1 erhöhen und vergleichen.
4. Merkmale mit `StandardScaler` skalieren, die lineare Regression neu trainieren und die Koeffizienten nach Betrag sortieren.

**Block 3: Logistische Regression auf dem Brustkrebs-Datensatz**

Achtung bei der Kodierung der Zielgröße: 0 = bösartig (malignant), 1 = gutartig (benign).

1. Die Kodierung prüfen (`target_names`, `np.bincount`), die Daten teilen und danach mit `StandardScaler` skalieren.
2. Verständnisfrage: Was ist falsch daran, erst alle Daten zu skalieren und danach zu teilen? Antwort als Text.
3. `LogisticRegression(max_iter=1000)` trainieren und Accuracy, Konfusionsmatrix und `classification_report` ausgeben.
4. Mit `predict_proba` die Schwelle 0.5 mit 0.3 vergleichen und die Tabelle für vier Schwellen bauen.

**Block 4: Entscheidungsbaum und Baumtiefe**

1. Einen Baum der Tiefe 2 trainieren und mit `export_text` ausgeben.
2. Bäume mit fünf verschiedenen Tiefen trainieren und Trainings- und Testgenauigkeit tabellieren.

**Block 5: Modellvergleich in einer Schleife**

1. Fünf Verfahren in einer Schleife trainieren und eine Tabelle mit Trainings- und Testgenauigkeit ausgeben.
2. Je Modell die Zahl der übersehenen bösartigen Proben und den Recall für die Klasse bösartig ergänzen (`pos_label=0`).
3. Verständnisfrage: Wie viele Testfälle entsprechen dem Abstand zwischen bestem und schwächstem Modell?

**Zusatzaufgaben**

1. `LinearRegression` mit einem, drei und allen acht Merkmalen: R² auf den Testdaten tabellieren.
2. Ein Streudiagramm Wahrheit gegen Vorhersage zeichnen und die Deckelung der Zielgröße suchen.
3. `random_state` im Split ändern und den Modellvergleich wiederholen.

## Lab 6: Feature Engineering, Evaluation und Tuning

Datei: `lab_06_evaluation_tuning.ipynb`, gehört zu Teil 6: Modelltraining, Feature Engineering, Hyperparameter und Evaluation.

**Lernziele**

+ Neue Merkmale bilden, schiefe Größen logarithmieren und Kategorien mit One-Hot kodieren
+ Overfitting am Abstand zwischen Trainingsfehler und Fehler auf neuen Daten erkennen
+ Accuracy, Precision, Recall, F1 sowie MAE, MSE, RMSE und R² von Hand rechnen und mit scikit-learn prüfen
+ Modelle mit Cross-Validation prüfen und Hyperparameter mit `GridSearchCV` suchen
+ Bei ungleichen Klassen die passende Kennzahl wählen, Klassen gewichten und die ROC-Kurve lesen

In diesem Lab beginnt die Nummerierung der Aufgaben in jedem Block neu bei 1.

**Block 1: Feature Engineering auf Titanic**

1. `FamilySize` und `IsAlone` bilden und die Überlebensrate vergleichen.
2. `Altersgruppe` mit `pd.cut` bilden und die Überlebensrate je Gruppe ausgeben.
3. `Fare` mit `np.log1p` logarithmieren und die Histogramme vorher und nachher zeichnen.
4. `Sex` und `Embarked` mit `pd.get_dummies` kodieren, danach `Sex` und `Pclass` mit `OneHotEncoder`. Beide Wege mit einer unbekannten Kategorie testen.

**Block 2: Overfitting am Polynombeispiel**

1. Polynome vom Grad 1, 4 und 11 anpassen und den Fehler auf den Trainingspunkten und auf neuen Punkten ausgeben.
2. Beide Fehler für alle Grade von 1 bis 11 tabellieren und über dem Grad zeichnen.

**Block 3: Kennzahlen von Hand**

1. Accuracy, Precision, Recall und F1 aus den vier Zahlen einer Konfusionsmatrix ausrechnen.
2. Die Werte mit `sklearn.metrics` prüfen.
3. MAE, MSE, RMSE und R² für fünf Wertepaare in NumPy rechnen und mit scikit-learn prüfen.

**Block 4: Cross-Validation**

1. Einen Entscheidungsbaum mit `cross_val_score` und `StratifiedKFold` prüfen: fünf Scores, Mittel, Streuung.
2. Dasselbe mit einem Random Forest wiederholen.
3. Statt `StratifiedKFold` ein `KFold` ohne Mischen nehmen und je Fold den Anteil der Klasse 1 ausgeben.

**Block 5: GridSearchCV**

1. Die beste Tiefe für einen `DecisionTreeRegressor` auf California Housing suchen, `cv_results_` als Tabelle ausgeben, auf dem Testteil prüfen.
2. Für einen Random Forest auf dem Brustkrebs-Datensatz `max_depth` und `n_estimators` mit `scoring="f1"` suchen.

**Block 6: Ungleiche Klassen**

1. Ein Modell trainieren, das immer die Mehrheit vorhersagt, und Accuracy, Recall und Konfusionsmatrix ausgeben.
2. Eine logistische Regression ohne und mit `class_weight="balanced"` trainieren und Accuracy, Precision und Recall tabellieren.
3. Beim Modell ohne Gewichte die Schwelle auf 0.3 und 0.7 verschieben.
4. Die ROC-Kurven beider Modelle in ein Diagramm zeichnen und die AUC berechnen.

**Zusatzaufgaben**

1. Einen groben Ausreißer in die Vorhersagen aus Block 3 einbauen: Welche Kennzahl reagiert am stärksten?
2. In `versicherte.csv` prüfen, ob `leistungsausgaben_eur` ein Kandidat für die Log-Transformation ist.
3. Die Suche aus Block 5, Aufgabe 2, mit `RandomizedSearchCV` und `n_iter=5` wiederholen.

## Lab 7: Clustering und Segmentierung

Datei: `lab_07_clustering.ipynb`, gehört zu Teil 7: Unüberwachtes Lernen.

**Lernziele**

+ Sie clustern Daten mit `KMeans`, zeichnen die Zentren und sehen, was ein falsch gewähltes k anrichtet.
+ Sie wählen k mit der Ellbogen-Kurve (`inertia_`) und dem `silhouette_score`.
+ Sie segmentieren die Versichertendaten und zeigen, warum Sie vor dem Clustern skalieren müssen.
+ Sie beschreiben Segmente mit einer Profiltabelle, geben ihnen Namen und schätzen ehrlich ein, wie scharf die Gruppen getrennt sind.
+ Sie stellen vier Merkmale mit PCA in zwei Dimensionen dar und lesen `explained_variance_ratio_` und die Ladungen.

In diesem Lab beginnt die Nummerierung der Aufgaben in jedem Block neu bei 1. Weicht Ihre Zahl vom Kontrollergebnis ab, prüfen Sie zuerst `random_state=1` und `n_init=10`.

**Block 1: k-Means auf Punktwolken**

1. 600 Punkte um 4 fest vorgegebene Zentren erzeugen, mit `KMeans` clustern und Zentren, `inertia_` und Clustergrößen ausgeben.
2. Die Punkte nach Cluster färben und die Zentren als rote Kreuze zeichnen.
3. k absichtlich falsch wählen: das Ergebnis für k = 2, 3 und 6 nebeneinander zeichnen.

**Block 2: k wählen mit Ellbogen-Kurve und Silhouetten-Wert**

1. Die `inertia_` für k = 2 bis 8 berechnen und die Ellbogen-Kurve zeichnen.
2. Den `silhouette_score` für k = 2 bis 8 berechnen und zeichnen.
3. Das beste k per Code aus der Liste bestimmen.

**Block 3: Versichertendaten segmentieren, erst ohne, dann mit Skalierung**

Die Datei enthält doppelte Zeilen und fehlende Werte in `bmi`. `KMeans` verträgt kein `NaN`. Eine vorbereitete Zelle zeigt die Fehlermeldung, damit Sie sie wiedererkennen.

1. `versicherte.csv` laden, Duplikate entfernen, danach Zeilen mit fehlenden Werten in den vier Merkmalen entfernen.
2. Die Größenordnungen der vier Merkmale mit `describe()` vergleichen.
3. Ohne Skalierung mit k = 4 clustern und Segmentgrößen und Mittelwerte je Segment ausgeben.
4. Mit `StandardScaler` skalieren, erneut clustern und beide Einteilungen mit `pd.crosstab` vergleichen.

**Block 4: Segmente beschreiben und ehrlich einordnen**

1. Den `silhouette_score` auf den skalierten Versichertendaten für k = 2 bis 8 berechnen: Wie scharf sind die Gruppen getrennt?
2. Die Profiltabelle bilden: Mittelwerte je Segment, Spalte `anzahl`, zum Vergleich der Gesamtdurchschnitt.
3. Raucheranteil und Anteil mit Zusatzversicherung je Segment berechnen.
4. Jedem Segment einen beschreibenden Namen geben und die Namen auszählen.

**Block 5: PCA auf zwei Komponenten**

1. Die skalierten Daten mit PCA auf zwei Spalten reduzieren und `explained_variance_ratio_` ausgeben.
2. Das Streudiagramm der beiden Hauptkomponenten zeichnen, gefärbt nach Segment.
3. Die Ladungen lesen: Welche Merkmale stecken in welcher Hauptkomponente?

**Zusatzaufgaben**

1. DBSCAN auf den skalierten Punktwolken mit zwei Werten für `eps`.
2. Die schiefen Leistungsausgaben vor dem Skalieren logarithmieren und erneut clustern.
3. Überlappende Punktwolken mit `n_init=1` und fünf verschiedenen Startwerten clustern und einen schlechten neben einem guten Lauf zeichnen.

## Lab 8: Einführung in Deep Learning mit PyTorch

Datei: `lab_08_deep_learning.ipynb`, gehört zu Teil 8: Einführung in Deep Learning.

**Lernziele**

+ Sie legen Tensoren an, rechnen mit ihnen, ändern ihre Form und tauschen Daten mit NumPy aus.
+ Sie berechnen Gradienten mit Autograd und prüfen das Ergebnis von Hand.
+ Sie laden FashionMNIST, sehen sich Bilder an und lesen die Form eines Batches.
+ Sie schreiben ein Netz als `nn.Module`, vervollständigen die Trainingsschleife und trainieren drei Epochen.
+ Sie messen die Genauigkeit auf den Testdaten, finden die verwechselten Klassen und speichern das Modell mit `state_dict`.

Alles läuft auf der CPU. In diesem Lab beginnt die Nummerierung der Aufgaben in jedem Block neu bei 1. Die Kontrollergebnisse gelten für `torch.manual_seed(42)` an den angegebenen Stellen. Kleine Abweichungen in der letzten Stelle sind je nach Rechner normal.

**Block 1: Tensoren**

1. Zwei Matrizen als Tensoren anlegen, Summe, elementweises Produkt und Matrizenprodukt berechnen, Form und Datentyp prüfen.
2. Die Form ändern: `arange`, `reshape` zu einer Matrix und zu einem Stapel von „Bildern", danach `nn.Flatten()`.
3. Ein NumPy-Array in einen Tensor umwandeln, auf `float32` umstellen und zurück in ein NumPy-Array wandeln.

**Block 2: Autograd**

1. Nachmachen: den Gradienten von `y = x**2 + 3*x + 5` bei x = 2.0 berechnen.
2. Abwandeln: den Gradienten von `y = 3*x**2 - 4*x` bei x = 1.0 berechnen, Handrechnung als Kommentar.
3. Selbst lösen: für ein Neuron mit drei Eingaben den Verlust und die Gradienten nach Gewichten und Bias berechnen und von Hand prüfen.

**Block 3: FashionMNIST laden und ansehen**

1. Trainings- und Testdaten mit `transform=ToTensor()` laden. Wichtig ist `root=str(DATA)`.
2. Das erste Trainingsbild mit Label holen, Form und Pixelwerte ausgeben und das Bild mit deutschem Klassennamen zeigen.
3. `train_loader` und `test_loader` mit `batch_size=64` anlegen und Zahl der Batches und Form eines Batches ausgeben.

**Block 4: Das Netz als `nn.Module`**

1. Ein lauffähiges Netz mit einer Linearschicht auf zwei Schichten (784 auf 128 auf 10) mit ReLU umbauen und die Parameter zählen.
2. Die Zahl der Parameter von Hand nachrechnen und je Schicht ausgeben.
3. Einen Batch durch das untrainierte Netz schicken: Form der Ausgabe und vorhergesagte Klassen ansehen.

**Block 5: Trainingsschleife**

1. In der Funktion `trainiere` drei fehlende Zeilen ergänzen und drei Epochen mit Adam trainieren.
2. Den Verlust je Epoche zeichnen und die Zahl der Schritte des Optimierers ausrechnen.
3. Lernrate ausprobieren: je ein frisches Netz eine Epoche lang mit `lr=1e-1` und mit `lr=1e-5` trainieren.

**Block 6: Auswerten, Verwechslungen ansehen, speichern**

1. Die Genauigkeit des trainierten Modells auf den Testdaten berechnen.
2. Die Verwechslungstabelle und die Genauigkeit je Klasse bilden: Welche Klasse ist am schwierigsten, womit wird sie verwechselt?
3. Die Gewichte mit `state_dict` speichern, in ein neues Objekt laden und die Genauigkeit vergleichen.

**Zusatzaufgaben**

1. Breitere Schicht: `NetzFlex(hidden=256)` drei Epochen trainieren und mit dem Netz aus Block 5 vergleichen.
2. Dropout: `NetzFlex(hidden=128, dropout=0.3)` drei Epochen trainieren und vergleichen.
3. SGD statt Adam: ein frisches Netz drei Epochen mit `optim.SGD` trainieren.
4. Nur mit Netzverbindung: dieselben Bilder über Hugging Face `datasets` laden. Ohne Netz überspringen Sie die Aufgabe.

## Lab 9: Der ML-Workflow als Pipeline

Datei: `lab_09_workflow_pipeline.ipynb`, gehört zu Teil 9: Der ML-Workflow mit scikit-learn.

**Lernziele**

+ Sie erzeugen ein Datenleck selbst und sehen, welche Größen dabei aus den Testdaten ins Training gelangen.
+ Sie bauen eine `Pipeline` aus Aufbereitung und Modell und greifen auf einzelne Schritte zu.
+ Sie behandeln Zahlen- und Kategoriespalten mit `ColumnTransformer` getrennt.
+ Sie bewerten die ganze Pipeline mit `cross_val_score` und stellen sie mit `GridSearchCV` ein.
+ Sie speichern die fertige Pipeline mit `joblib`, laden sie wieder und sagen neue Passagiere vorher.

In diesem Lab beginnt die Nummerierung der Aufgaben in jedem Block neu bei 1. Die Kontrollergebnisse gelten für `random_state=1`. Ab Block 3 bauen Sie ein Skript auf, das bis zum Ende des Labs weiterläuft: Spätere Blöcke brauchen die Ergebnisse der früheren.

**Block 1: Das Datenleck selbst erzeugen**

1. Mit Leck: auf allen Zeilen füllen und skalieren, erst danach teilen, logistische Regression trainieren.
2. Ohne Leck: zuerst teilen, Imputer und Scaler lernen nur auf den Trainingsdaten.
3. Vergleichen, was Imputer und Scaler in beiden Fassungen gelernt haben.

**Block 2: Erste Pipeline mit Zahlenspalten**

1. Eine Pipeline aus `StandardScaler` und `LogisticRegression` bauen und mit den Spalten ohne Lücken trainieren.
2. Über `named_steps` die gelernten Mittelwerte auslesen, mit `set_params` den Parameter `C` umstellen, neu trainieren.
3. `Age` dazunehmen: den Fehler mit `try/except` abfangen und einen `SimpleImputer` als ersten Schritt ergänzen.

**Block 3: ColumnTransformer für Zahlen und Kategorien**

1. `num_cols` und `cat_cols` festlegen, `X` und `y` bilden und stratifiziert teilen.
2. `numeric_pipe`, `categorical_pipe` und daraus `preprocess` bauen, Form und Spaltennamen ansehen.
3. `preprocess` und die logistische Regression zur Pipeline `clf` zusammensetzen und trainieren.

**Block 4: Modell tauschen und mit Cross-Validation bewerten**

1. `clf` mit `cross_val_score` bewerten und Mittelwert und Streuung ausgeben.
2. Die Pipeline `rf` mit einem Random Forest bauen und genauso bewerten.
3. Ein drittes Modell wählen und alle drei Ergebnisse in einer Tabelle zusammenstellen.

**Block 5: GridSearchCV über Pipeline-Parameter**

1. Mit `GridSearchCV` über `modell__max_depth` und `modell__n_estimators` suchen.
2. `cv_results_` als Tabelle lesen, sortiert nach `rank_test_score`.
3. Das beste Modell einmal auf dem Testset auswerten: Konfusionsmatrix und `classification_report`.

**Block 6: Speichern, laden, vorhersagen**

1. Die beste Pipeline mit `joblib.dump` speichern, wieder laden und die Accuracy auf dem Testset prüfen.
2. Einen neuen Passagier als DataFrame mit einer Zeile anlegen, vorhersagen, danach Geschlecht und Klasse ändern.
3. Eine Passagierin mit fehlendem Alter (`np.nan`) und unbekanntem Hafen vorhersagen und begründen, warum kein Fehler kommt.

**Zusatzaufgaben**

1. Die Strategie des Imputers in die Gittersuche aufnehmen.
2. Aus `Name` den Titel als zusätzliche Kategoriespalte gewinnen und per Cross-Validation prüfen, ob das Modell besser wird.
3. Ein Leck mit großer Wirkung: Merkmalsauswahl auf reinen Zufallszahlen, einmal vor der Cross-Validation, einmal als Schritt der Pipeline.

## Lab 10: Praxisprojekt und Interpretation

Datei: `lab_10_praxisprojekt.ipynb`, gehört zu Teil 10: Praxisbeispiele und Interpretation von Modellen.

**Lernziele**

+ Sie wählen Merkmale so, dass keine Spalte die Antwort schon enthält.
+ Sie bewerten ein Modell bei stark ungleichen Klassen mit Recall und Precision und legen eine Schwelle fest.
+ Sie vergleichen `feature_importances_` mit `permutation_importance` und lesen einen Entscheidungsbaum.
+ Sie lesen die Koeffizienten eines linearen Modells nach dem Skalieren und erkennen im Residuenplot eine gedeckelte Zielgröße.
+ Sie führen ein eigenes Mini-Projekt von der Frage bis zu drei Sätzen für die Fachabteilung durch.

Teil A (Blöcke 1 bis 3) ist geführt, mit Kontrollergebnissen zu jeder Aufgabe. Teil B (Block 4) ist Ihr eigenes Projekt auf einem Datensatz nach Wahl. In Teil A beginnt die Nummerierung der Aufgaben in jedem Block neu bei 1.

**Block 1: Maschinenausfall vorhersagen**

1. Form und Zielgröße ansehen: Wie viele Ausfälle gibt es, welche Accuracy erreicht ein Modell ohne jedes Lernen?
2. Verständnisaufgabe: die Merkmale wählen. Prüffrage für jede Spalte: „Kenne ich diesen Wert, bevor der Ausfall eintritt?" Danach teilen und die Aufbereitung `prep_m` bauen.
3. Mit der vorbereiteten Funktion `bewerte` eine Basislinie und einen Random Forest mit `class_weight="balanced"` vergleichen.
4. Die Schwelle über Cross-Validation wählen und das Testset einmal mit Schwelle 0.3 auswerten.

**Block 2: Was hat das Modell gelernt?**

1. Einen Random Forest auf Titanic mit einer zusätzlichen Zufallsspalte trainieren und `feature_importances_` sortiert ausgeben.
2. `permutation_importance` auf den Testdaten berechnen und wieder sortieren.
3. Mit `plot_tree` einen Entscheidungsbaum der Tiefe 3 zeichnen und eine Regel in einem Satz formulieren.

**Block 3: Regression auf California Housing**

1. Teilen, eine Pipeline aus `StandardScaler` und `LinearRegression` bauen, RMSE per Cross-Validation und auf dem Testset bestimmen, dazu R².
2. Die Koeffizienten als Tabelle lesen, sortiert nach Betrag.
3. Den Residuenplot zeichnen und die Deckelung der Zielgröße suchen.

**Block 4: Von der Frage bis zum Ergebnis für die Fachabteilung**

Zur Wahl stehen `titanic.csv` (Wer überlebt?), `california_housing.csv` (Wie hoch ist der Hauswert eines Bezirks?), `winequality-red.csv` (Ist ein Wein gut? Oder: Welche Note bekommt er?) und `versicherte.csv` (Welche Gruppen von Versicherten gibt es?). Jede Zelle ist ein Schritt. Bei einer Segmentierung entfallen Split, Basislinie und Testauswertung, stattdessen berechnen und benennen Sie ein Segmentprofil.

1. Frage formulieren: Welche Entscheidung soll das Modell unterstützen, welche Kennzahl passt dazu?
2. Daten ansehen: Form, Datentypen, fehlende Werte, Duplikate, Verteilung der Zielgröße.
3. Zielgröße und Merkmale festlegen: Kennungen weglassen, Prüffrage für jede Spalte stellen.
4. Split: `random_state=1`, bei Klassifikation `stratify=y`. Das Testset zur Seite legen.
5. Basislinie: `DummyClassifier` oder `DummyRegressor`.
6. Pipeline: Aufbereitung und Modell in einem Objekt, am besten zwei Kandidaten.
7. Cross-Validation: Kandidaten auf den Trainingsdaten vergleichen, einen auswählen.
8. Testauswertung: genau einmal, mit Konfusionsmatrix oder Residuenplot.
9. Interpretation: `permutation_importance` auf den Testdaten, Richtung aus Koeffizienten.
10. Drei Sätze für die Fachabteilung: Wie gut ist es? Worauf stützt es sich? Wo darf man ihm nicht trauen?

**Zusatzaufgaben**

1. Das Leck absichtlich einbauen: den Random Forest aus Block 1 zusätzlich mit den Spalten der Ausfallarten trainieren und erklären, warum das Ergebnis wertlos ist.
2. Permutation Importance passend zur Frage: für das Ausfallmodell mit `scoring="recall"` berechnen.
3. Deckelung herausnehmen: alle gedeckelten Bezirke entfernen, die Pipeline neu trainieren und den RMSE vergleichen.
