import sqlite3
import pandas as pd

connection = sqlite3.connect("github.db")
df = pd.read_sql("SELECT * FROM repos", connection)
connection.close()

print("First rows:")
print(df.head())

print("\nShape (rows, columns):", df.shape)

print("\nRepos with a known language, sorted by name:")
print(df[df["language"].notna()].sort_values("name")[["name", "language"]])

print("\nCount by language:")
print(df.groupby("language").size().sort_values(ascending=False))
