"""Seed a FIXED, known database for CI evals — decoupled from live data.

The eval's ground truth lives HERE, not in the production github.db. CI runs
this before launching the API (with DB_FILE=fixture.db), so refreshing the real
repo data can never change the numbers the tests assert against.

Deliberately fake names (alpha, beta, ...) so fixture output is never mistaken
for real repos. Shape: 5 repos — 3 Python, 1 TypeScript, 1 CSS; alpha is the
single most-starred (100). Those facts are what promptfooconfig.yaml checks.
"""
import os
import sqlite3

DB_FILE = os.environ.get("DB_FILE", "fixture.db")

REPOS = [
    ("alpha",   100, "Python",     "https://example.com/alpha"),
    ("beta",     50, "Python",     "https://example.com/beta"),
    ("epsilon",  25, "Python",     "https://example.com/epsilon"),
    ("gamma",    10, "TypeScript", "https://example.com/gamma"),
    ("delta",     0, "CSS",        "https://example.com/delta"),
]

conn = sqlite3.connect(DB_FILE)
conn.execute("DROP TABLE IF EXISTS repos")
conn.execute(
    "CREATE TABLE repos (name TEXT, stars INTEGER, language TEXT, url TEXT)")
conn.executemany(
    "INSERT INTO repos (name, stars, language, url) VALUES (?, ?, ?, ?)", REPOS)
conn.commit()
conn.close()
print(f"Seeded {len(REPOS)} fixture repos into {DB_FILE}")
