# GitHub Repos API

A FastAPI service that serves GitHub repository data from a local SQLite database,
with a pandas-powered stats endpoint, LLM endpoints (pitch, structured analysis),
and a **tool-using agent** that answers questions by calling functions over the
data. Instrumented end to end with Langfuse and wrapped in input/output guardrails.

Built as an AI-engineering portfolio project demonstrating the full production
lifecycle: evaluation, observability, guardrails, and deployment.

## Live demo

**Live:** https://github-repos-api-lbg6.onrender.com

- Interactive API docs: https://github-repos-api-lbg6.onrender.com/docs
- Ask the agent: https://github-repos-api-lbg6.onrender.com/agent?q=How many Python repositories do I have?

> Hosted on Render's free tier; the first request after an idle period takes
> ~30-60s to wake.

## Features

- REST API with FastAPI, interactive Swagger docs at `/docs`
- Data served from SQLite with parameterized queries; language stats via pandas
- A tool-using `/agent` endpoint (LLM function calling over the database)
- Structured analysis (`/repos/{name}/analyze`) validated against a Pydantic schema
- Groq (`openai/gpt-oss-20b`) for all generation

## Endpoints

| Method | Path                    | Description                                   |
| ------ | ----------------------- | --------------------------------------------- |
| GET    | `/repos`                | List all repos (optional `?language=` filter) |
| GET    | `/repos/{name}`         | Get a single repo by name                     |
| GET    | `/repos/{name}/pitch`   | LLM-generated one-line pitch                  |
| GET    | `/repos/{name}/analyze` | Schema-validated structured analysis          |
| GET    | `/agent`                | Tool-using agent; answers a `?q=` question    |
| GET    | `/stats`                | Repo counts by language (pandas)              |

## Observability

Every LLM call is traced with Langfuse (drop-in OpenAI wrapper). The `/agent`
handler opens a parent span and wraps each tool call, so one request is a single
trace showing the full loop:

```
agent
├── guardrail:injection   (input screening)
├── generation            (model decides which tool to call)
├── tool:count_repos      (DB query, with arguments and result)
└── generation            (model writes the final answer)
```

Prompt, response, latency, tokens, and cost are captured per step, with a custom
model price registered so Groq costs are calculated.

## Guardrails

- **Input, injection gate:** every `/agent` question is screened by a policy
  classifier before any model call or tool runs. This matters because the agent has
  live database tools, a successful injection is a real risk, not just bad text.
  Blocked requests never reach the database.
- **Output, schema enforcement:** `run_analysis` forces the model's output into a
  Pydantic `RepoAnalysis` schema, retrying with the validation error fed back, and
  falling back gracefully if it can't produce a valid result.
- **Graceful fallback:** a model or tool failure returns a safe degraded response
  (HTTP 200), never a 500 stack trace.

## Results

- The agent trace shows the full reason-act-observe-answer loop; injection attempts
  are stopped at the guard with no tool execution (verifiable in the trace).
- Two model calls per answer make the agent ~2x the cost/latency of a single-shot
  endpoint, quantified from the traces rather than guessed.
- Structured analysis is schema-valid or gracefully degraded, never malformed.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```

Environment variables (git-ignored `.env` locally; host secrets in production):

```
GROQ_API_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

Requires a `github.db` SQLite file with a `repos` table (`name`, `stars`,
`language`, `url`); it is committed so it deploys with the app.

## Deployment

Runs as a Python web service on Render's free tier. Lightweight (no ML libraries),
so it fits the 512MB limit easily. Secrets are Render environment variables; the
SQLite database travels in the repo. Start command:
`uvicorn main:app --host 0.0.0.0 --port $PORT`.

## What it demonstrates

Building a REST API with FastAPI and Pydantic; SQLite with parameterized queries;
data aggregation with pandas; an LLM tool-using agent loop; output-schema
enforcement; distributed tracing (Langfuse) for latency/token/cost; input and
output guardrails; and free-tier cloud deployment with managed secrets.
