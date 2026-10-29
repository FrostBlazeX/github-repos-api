# GitHub Repos API

A small FastAPI service that serves GitHub repository data from a local SQLite
database, with a language-stats endpoint powered by pandas, an AI endpoint
that uses an LLM to generate a one-line pitch for any repo, and a tool-using
agent that answers questions about the repos. Every LLM call is traced with
Langfuse for latency, token, and cost visibility.

Built as a Python + AI-engineering learning project.

## Features

- REST API built with FastAPI, with interactive Swagger docs at `/docs`
- Data served from a local SQLite database (`github.db`)
- Typed request/response models with Pydantic
- Path and query parameters, plus proper 404 handling
- A pandas-powered `/stats` endpoint (repo counts by language)
- An LLM-powered `/repos/{name}/pitch` endpoint (via Groq), with the API key
  kept out of the code in a `.env` file
- A tool-using `/agent` endpoint that answers questions by calling functions
  over the SQLite data (LLM function calling), never guessing numbers or names
- Full request tracing with Langfuse: prompt, response, latency, tokens, and
  cost captured for every LLM call, grouped into one trace per request

## Requirements

- Python 3.12+
- A free Groq API key from https://console.groq.com (needed for `/pitch` and `/agent`)
- A free Langfuse account from https://cloud.langfuse.com (for tracing)
- Dependencies listed in `requirements.txt`

## Setup

    python -m venv .venv
    .venv\Scripts\Activate.ps1        # Windows (PowerShell)
    # source .venv/bin/activate       # macOS / Linux
    python -m pip install -r requirements.txt

Create a `.env` file in the project root with your Groq key and Langfuse keys:

    GROQ_API_KEY=your_key_here
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=https://cloud.langfuse.com      # us.cloud.langfuse.com for the US region

(`.env` is git-ignored, so the keys are never committed.)

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
| GET    | `/agent`              | Tool-using agent; answers a `?q=` question    |
| GET    | `/stats`              | Repo counts grouped by language (via pandas)  |

## Observability

Every LLM call in the service is traced with [Langfuse](https://langfuse.com/).
The OpenAI client is swapped for Langfuse's drop-in wrapper
(`from langfuse.openai import OpenAI`), so each `chat.completions.create` call is
recorded automatically with its prompt, response, latency, and token usage. The
`/agent` handler opens an explicit parent span and wraps each tool call in its own
child span, so one request becomes a single trace showing the full loop:

    agent
    ├── generation      (model decides which tool to call)
    ├── tool:count_repos  (DB query, with its arguments and result)
    └── generation      (model writes the final answer)

This makes the agent debuggable: you can see exactly which tool the model chose,
what arguments it passed, and what came back, rather than inferring it from a wrong
answer.

Cost is enabled by registering a custom model price for the Groq model in Langfuse
(Project Settings → Models), since Groq pricing is not built in. With that in place,
each generation shows its cost, which makes the per-request economics visible (for
example, the agent's two model calls cost roughly twice a single-call endpoint).

To view traces, make a request and open your Langfuse project under Tracing → Traces.

## What it demonstrates

- Building a REST API with FastAPI and Pydantic
- Serving data from SQLite with parameterized queries
- Data aggregation with pandas
- Calling an LLM from a backend service
- LLM function calling / a tool-using agent loop
- Instrumenting an LLM service with tracing (Langfuse): latency, token, and cost
  visibility across multi-step flows
- Managing secrets with environment variables (`python-dotenv`)
