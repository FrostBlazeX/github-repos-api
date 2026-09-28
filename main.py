import os
import sqlite3
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DB_FILE = "github.db"

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

app = FastAPI(title="GitHub Repos API")


class Repo(BaseModel):
    name: str
    stars: int
    language: str | None
    url: str


def query_db(sql, params=()):
    connection = sqlite3.connect(DB_FILE)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    connection.close()
    return [dict(row) for row in rows]


@app.get("/")
def home():
    return {"message": "GitHub Repos API — visit /docs to explore."}


@app.get("/repos", response_model=list[Repo])
def list_repos(language: str | None = None):
    if language:
        return query_db("SELECT * FROM repos WHERE language = ? COLLATE NOCASE", (language,))
    return query_db("SELECT * FROM repos")


@app.get("/repos/{name}", response_model=Repo)
def get_repo(name: str):
    results = query_db("SELECT * FROM repos WHERE name = ?", (name,))
    if not results:
        raise HTTPException(status_code=404, detail=f"No repo named '{name}'")
    return results[0]


@app.get("/repos/{name}/pitch")
def pitch_repo(name: str):
    results = query_db("SELECT * FROM repos WHERE name = ?", (name,))
    if not results:
        raise HTTPException(status_code=404, detail=f"No repo named '{name}'")
    repo = results[0]

    prompt = (
        f"Write a single punchy sentence pitching this GitHub project to a recruiter. "
        f"Project name: {repo['name']}. Primary language: {repo['language'] or 'unknown'}. "
        f"Be concrete and enthusiastic, and stay under 25 words."
    )

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
    )
    pitch = completion.choices[0].message.content.strip()

    return {"name": repo["name"], "pitch": pitch}


@app.get("/stats")
def language_stats():
    connection = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM repos", connection)
    connection.close()
    return df.groupby("language").size().sort_values(ascending=False).to_dict()
