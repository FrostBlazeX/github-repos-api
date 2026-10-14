import os
import sqlite3
import pandas as pd
from fastapi import FastAPI, HTTPException
from typing import Literal
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from openai import OpenAI
import json


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


class RepoAnalysis(BaseModel):
    summary: str = Field(
        description="One or two sentences on what the project is.")
    suggested_topics: list[str] = Field(
        description="3-5 short GitHub topic tags.")
    difficulty: Literal["beginner", "intermediate", "advanced"]


def query_db(sql, params=()):
    connection = sqlite3.connect(DB_FILE)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    connection.close()
    return [dict(row) for row in rows]


def run_analysis(repo) -> RepoAnalysis:
    system_prompt = (
        "You analyze software repositories. "
        "Respond with a SINGLE JSON object and nothing else. "
        "Keys: 'summary' (string), 'suggested_topics' (array of 3-5 short strings), "
        "'difficulty' (exactly one of: 'beginner', 'intermediate', 'advanced')."
    )
    user_prompt = (
        f"Analyze this GitHub repository.\n"
        f"Name: {repo['name']}\n"
        f"Primary language: {repo['language'] or 'unknown'}\n"
        f"URL: {repo['url']}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_error = None
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            raw = completion.choices[0].message.content
            return RepoAnalysis.model_validate_json(raw)
        except ValidationError as e:
            last_error = e
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                             f"That did not match the schema. Error: {e}. "
                             f"Return corrected JSON with exactly the required keys and types."})
        except Exception as e:
            last_error = e

    raise HTTPException(status_code=502,
                        detail=f"Could not analyze after 3 attempts: {last_error}")


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


@app.get("/repos/{name}/analyze", response_model=RepoAnalysis)
def analyze_repo(name: str):
    results = query_db("SELECT * FROM repos WHERE name = ?", (name,))
    if not results:
        raise HTTPException(status_code=404, detail=f"No repo named '{name}'")
    return run_analysis(results[0])


@app.get("/analyze-all")
def analyze_all():
    repos = query_db("SELECT * FROM repos")
    output = []
    for repo in repos:
        try:
            output.append(
                {"name": repo["name"], "analysis": run_analysis(repo)})
        except Exception as e:
            output.append({"name": repo["name"], "analysis": None,
                           "error": type(e).__name__ + ": " + str(e)})
    return output


@app.get("/stats")
def language_stats():
    connection = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM repos", connection)
    connection.close()
    return df.groupby("language").size().sort_values(ascending=False).to_dict()
