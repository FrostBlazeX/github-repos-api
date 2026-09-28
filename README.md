# GitHub Repos API

A small FastAPI service that serves GitHub repository data from a local SQLite
database, with a language-stats endpoint powered by pandas and an AI endpoint
that uses an LLM to generate a one-line pitch for any repo.

Built as a Python + AI-engineering learning project.

## Features

- REST API built with FastAPI, with interactive Swagger docs at `/docs`
- Data served from a local SQLite database (`github.db`)
- Typed request/response models with Pydantic
- Path and query parameters, plus proper 404 handling
- A pandas-powered `/stats` endpoint (repo counts by language)
- An LLM-powered `/repos/{name}/pitch` endpoint (via Groq), with the API key
  kept out of the code in a `.env` file

## Requirements

- Python 3.12+
- A free Groq API key from https://console.groq.com (only needed for `/pitch`)
- Dependencies listed in `requirements.txt`

## Setup

    python -m venv .venv
    .venv\Scripts\Activate.ps1        # Windows (PowerShell)
    # source .venv/bin/activate       # macOS / Linux
    python -m pip install -r requirements.txt

Create a `.env` file in the project root with your Groq key:

    GROQ_API_KEY=your_key_here

(`.env` is git-ignored, so the key is never committed.)

You also need a `github.db` SQLite database with a `repos` table
(columns: `name`, `stars`, `language`, `url`). It was built alongside a
companion script that populates it from the GitHub API.

## Running

    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs to explore the API interactively.

## Endpoints

| Method | Path                  | Description                                   |
| ------ | --------------------- | --------------------------------------------- |
| GET    | `/`                   | Welcome message                               |
| GET    | `/repos`              | List all repos (optional `?language=` filter) |
| GET    | `/repos/{name}`       | Get a single repo by name                     |
| GET    | `/repos/{name}/pitch` | LLM-generated one-line pitch for a repo       |
| GET    | `/stats`              | Repo counts grouped by language (via pandas)  |

## What it demonstrates

- Building a REST API with FastAPI and Pydantic
- Serving data from SQLite with parameterized queries
- Data aggregation with pandas
- Calling an LLM from a backend service
- Managing secrets with environment variables (`python-dotenv`)
