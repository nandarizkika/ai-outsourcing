# AI Talent

A multi-agent AI analytics service built on FastAPI. Accepts natural language requests, routes them through specialized agents, and delivers results via Slack, email, or API response.

## What it does

- Answers data and business questions in natural language (`/analyze`)
- Connects to your database (PostgreSQL, MySQL, BigQuery, Snowflake) and runs SQL queries
- Generates charts, reports, ML forecasts, A/B test analyses, segmentation, and anomaly detection
- Delivers scheduled reports via Slack or email
- Tracks per-client API keys, request tracing, and structured JSON logs

## Requirements

- Python 3.11+
- Anthropic API key
- OpenAI API key
- Slack app (bot token + signing secret)

## Setup

**1. Clone and install**

```bash
git clone <your-repo-url>
cd ai_talent
python3.11 -m venv .venv
.venv/bin/pip install -e .
```

**2. Configure environment**

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | From console.anthropic.com |
| `OPENAI_API_KEY` | Yes | From platform.openai.com |
| `SLACK_BOT_TOKEN` | Yes | `xoxb-...` from your Slack app |
| `SLACK_SIGNING_SECRET` | Yes | From your Slack app settings |
| `API_KEY` | Yes | Any secret string — admin key for management routes |
| `CHROMADB_PERSIST_DIR` | No | Default: `.chromadb` |
| `JIRA_URL` | No | Enables Jira ticketing integration |
| `EMAIL_IMAP_HOST` | No | Enables email channel |

**3. Start the server**

```bash
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
```

Logs are emitted as structured JSON to stdout.

## Quick start

**Register a client**

```bash
curl -X POST http://localhost:8000/clients \
  -H "X-API-Key: <your-admin-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "acme",
    "name": "Acme Corp",
    "tier": "basic",
    "enabled_skills": [],
    "account_mode": "vendor",
    "active_channels": ["slack"]
  }'
```

The response includes an `api_key` — save it, it is only shown once.

```json
{"client_id": "acme", "api_key": "a1b2c3d4..."}
```

**Analyze a request**

```bash
curl -X POST http://localhost:8000/analyze \
  -H "X-Client-ID: acme" \
  -H "X-API-Key: <client-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "request": {
      "channel": "slack",
      "sender_id": "U123",
      "sender_name": "Alice",
      "text": "Show me revenue trends for last quarter",
      "timestamp": "2026-05-22T00:00:00Z",
      "client_id": "acme"
    },
    "clarification_state": null
  }'
```

## API reference

### Auth

| Route | Auth |
|---|---|
| `/clients/*` | `X-API-Key: <admin-key>` |
| `/jobs/*` | `X-API-Key: <admin-key>` |
| `/analyze` | `X-Client-ID: <id>` + `X-API-Key: <client-key>` |
| `/health` | None |

### Clients

| Method | Route | Description |
|---|---|---|
| `POST` | `/clients` | Register a client, returns `api_key` |
| `GET` | `/clients` | List all clients |
| `GET` | `/clients/{id}` | Get a client |
| `DELETE` | `/clients/{id}` | Delete a client |
| `POST` | `/clients/{id}/rotate-key` | Rotate the client's API key |

### Analysis

| Method | Route | Description |
|---|---|---|
| `POST` | `/analyze` | Submit a natural language request |

### Scheduled jobs

| Method | Route | Description |
|---|---|---|
| `POST` | `/jobs` | Create a scheduled job (cron) |
| `GET` | `/jobs` | List scheduled jobs |
| `DELETE` | `/jobs/{id}` | Remove a scheduled job |

### Other

| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/slack/events` | Slack event webhook |

## Connecting a database

Pass `connector_type` and `connector_config` when registering a client:

**PostgreSQL / MySQL**
```json
{
  "client_id": "acme",
  "connector_type": "postgres",
  "connector_config": {
    "host": "db.example.com",
    "database": "analytics",
    "user": "readonly",
    "password": "secret"
  }
}
```

**BigQuery** (uses Application Default Credentials — run `gcloud auth application-default login` first)
```json
{
  "connector_type": "bigquery",
  "connector_config": {
    "project_id": "my-gcp-project",
    "dataset": "analytics"
  }
}
```

**Snowflake**
```json
{
  "connector_type": "snowflake",
  "connector_config": {
    "account": "myaccount",
    "user": "admin",
    "password": "secret",
    "warehouse": "COMPUTE_WH",
    "database": "ANALYTICS"
  }
}
```

Rotate a client's API key:
```bash
curl -X POST http://localhost:8000/clients/acme/rotate-key \
  -H "X-API-Key: <your-admin-key>"
```

## Deployment

**Docker (recommended)**

```bash
docker compose up -d
```

The app runs on port 8000. Mount `clients.json` and `.chromadb` as volumes so data survives restarts (see `docker-compose.yml`).

**HTTPS with Caddy**

[Caddy](https://caddyserver.com) handles TLS certificates automatically via Let's Encrypt. Install it, edit `Caddyfile` to replace `your-domain.com` with your actual domain, then:

```bash
caddy run
```

Caddy proxies HTTPS → `localhost:8000`. Your app needs no TLS config changes.

**Rate limiting**

`/analyze` is limited to **60 requests per minute per client** (keyed on `X-Client-ID`). Clients that exceed this receive a `429 Too Many Requests` response.

## Running tests

```bash
.venv/bin/python -m pytest
```

269 tests across unit, integration, and HTTP layers.

## Architecture

```
main.py                  FastAPI app, route handlers, middleware wiring
src/
  core/                  Config, models, auth, client registry, logging
  agents/                Specialized agents (SQL, ML, chart, report, ...)
  orchestrator/          Routes requests to agents, manages clarification
  connectors/            Database connectors (SQLAlchemy-based)
  channels/              Slack and email delivery
  jobs/                  APScheduler-based scheduled jobs
  knowledge/             ChromaDB vector store + retriever
  middleware/            Request logging middleware (X-Request-ID tracing)
  ticketing/             Jira integration
```

Each request to `/analyze` is logged with a unique `X-Request-ID` that appears in every log line for that request, making it easy to trace across agents.
