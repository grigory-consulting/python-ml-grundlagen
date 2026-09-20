"""Erzeugt data/versicherte.db (SQLite) aus data/versicherte.csv für die Folie zu pd.read_sql."""
import pathlib
import sqlite3

import pandas as pd

here = pathlib.Path(__file__).parent
df = pd.read_csv(here / "versicherte.csv", dtype={"plz": str})

db = here / "versicherte.db"
db.unlink(missing_ok=True)
with sqlite3.connect(db) as con:
    df.to_sql("versicherte", con, index=False)
print(f"{db.name}: {len(df)} Zeilen in Tabelle 'versicherte'")
