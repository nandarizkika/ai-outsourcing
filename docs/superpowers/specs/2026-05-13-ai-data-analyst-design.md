# AI Data Analyst — Design Spec

**Date:** 2026-05-13
**Project:** AI Talent Outsourcing
**Status:** Approved

---

## 1. Concept

The AI Data Analyst is a digital employee — not a tool or chatbot. It has its own identity in company systems (email, Slack, WhatsApp, Jira), receives and responds to requests through those channels, manages its own workload, and delivers analysis just like a human analyst would.

Each client gets one dedicated AI employee instance, fully configured to their role, business domain, and data sources.

---

## 2. Core Capabilities

All three activate from day one, in deployment order:

| Capability | Description | Active From |
|---|---|---|
| On-demand analysis | Responds to requests from any channel | Day 1 |
| Hard-rule anomaly detection | Enforces client-defined business rules on every data run | Day 1 |
| Scheduled jobs | Recurring autonomous tasks (e.g. weekly reports) | Day 1 |
| Statistical anomaly detection | Pattern-based deviation detection from learned baseline | Day 30+ |
| ML model building | Forecasting, classification, clustering | On request |
| Presentation deck generation | Slide decks from analysis results | On request |

---

## 3. Architecture — Modular Orchestrator

```
CHANNEL LAYER (opt-in per client)
  Email · Slack · WhatsApp · Jira/Linear
       ↓ normalize to standard Request format
CHANNEL ADAPTER
  · Normalizes all incoming requests
  · Routes responses back to origin channel
  · Auto-creates Jira ticket from informal requests (Slack/WA/Email → Jira)
       ↓
ORCHESTRATOR AGENT
  1. Understand — parse intent, load relevant domain knowledge
  2. Validate Clarity — if ambiguous, ask back via same channel (max 2 rounds)
     → if still unclear after 2 rounds: proceed with best interpretation, state assumptions explicitly
  3. Plan — which sub-agents, what order, sync or async
  4. Acknowledge — reply to channel: "Got it, I'll have this ready by EOD. Logged as JIRA-204."
  5. Delegate — dispatch to sub-agents / job queue
       ↓ (powered by LLM Layer)
LLM LAYER (abstracted, multi-provider)
  Claude        → complex reasoning, long-context analysis
  GPT-4o        → tool-heavy tasks
  Haiku / Mini  → simple summarization, cheap tasks
  (provider swappable per task type — no vendor lock-in)
       ↓ delegates to
SUB-AGENTS
  SQL Agent · Sheets Agent · Chart Agent · Deck Agent · ML Agent · Anomaly Agent
       ↓ reads from
DATA CONNECTORS          KNOWLEDGE LAYER           JOB QUEUE
SQL · Sheets · Looker    Vector DB (per client)    Async long-running tasks
```

---

## 4. Channel Layer

### 4.1 Account Mode

Channels support two account modes, switchable at any time:

| Mode | Description | When to Use |
|---|---|---|
| **Vendor-managed** | AI uses the vendor's own accounts (e.g. dian@aiagent.com, shared Slack bot) | Pilot phase, early clients — fast setup, no client IT needed |
| **Client-dedicated** | AI gets its own client-specific accounts (e.g. dian@clientcompany.com) | Production deployments — professional, feels like a real employee |

### 4.2 Available Channels

All channels are opt-in. Client selects which to activate during onboarding. Ticketing is strongly recommended for all clients regardless of other channel choices.

| Channel | Identity | Integration |
|---|---|---|
| Email | dian@[domain] | IMAP / SMTP or Gmail API |
| Slack | @dian-ai | Slack Bot Token |
| WhatsApp | Dedicated WA Business number | WhatsApp Business API |
| Jira / Linear | Dian (AI Analyst) user account | Webhook + REST API |

Channels can be added or removed later via a change request.

If no ticketing system is activated: self-ticketing is disabled. The AI still functions via other channels but has no internal task tracking. Ticketing activation is strongly recommended.

### 4.3 Self-Ticketing

When a request arrives via an informal channel (Slack, WhatsApp, Email), the Channel Adapter automatically creates a Jira/Linear ticket:
- Title derived from the request
- Assigned to the AI employee
- Linked back to the originating thread/email
- AI references the ticket in its acknowledgment reply

### 4.4 Clarification Loop

```
Request received
  → Orchestrator validates clarity
  → If ambiguous: ask back via same channel (max 2 rounds)
  → If still unclear after 2 rounds:
       proceed with best interpretation
       state all assumptions explicitly in output
       never silently guess
```

---

## 5. Knowledge Layer

### 5.1 Isolation

Each client has a completely isolated knowledge base namespace in the vector DB. No cross-contamination between clients.

### 5.2 Two Sources

**Source 1 — Onboarding upload (static foundation)**
- Client uploads: SOPs, data dictionaries, glossaries, past reports, report templates, anomaly rule documents
- Pipeline: document → parse (PDF/DOCX/XLSX/CSV) → chunk → embed → store in client namespace
- Updates: client can upload new documents anytime via portal → ingested within 24 hours

**Source 2 — Interaction memory (continuous learning)**
- Every request received, query executed, output delivered, and correction given is logged as a structured memory entry
- Same embedding pipeline → appended to client namespace
- Over time: the AI learns preferred formats, metric definitions, business terminology, successful query patterns

### 5.3 Retrieval

On every incoming request, a semantic search retrieves the top-K most relevant chunks from the client's namespace and injects them into the Orchestrator's context as domain background.

### 5.4 What Gets Remembered

- Successful SQL queries → reused as templates
- Business term corrections ("when we say NPL, we mean 90+ days, not 30")
- Preferred output format per requestor
- Anomaly baselines (after 30 days)
- Recurring task patterns and schedules
- Feedback like "this metric should always exclude intercompany transactions"

---

## 6. Data Connectors

Configured per client during onboarding. All connections are read-only except Google Sheets (write to designated output tab only).

| Connector | Supported Sources | Notes |
|---|---|---|
| SQL | PostgreSQL, MySQL, BigQuery, MS SQL Server, Snowflake | Read-only credentials |
| Sheets | Google Sheets API, Excel file upload | Read + write to output tab |
| Looker | Looker API — Looks, Explores, LookML queries | Read-only (MVP) |

---

## 7. Skill Modules

Skill modules are discrete technical capabilities that can be enabled or disabled per client deployment. They map directly to sub-agents. During onboarding, your team selects which modules to activate based on the client's tier and scope of work.

The Orchestrator only delegates to enabled skill modules — it will not invoke a sub-agent for a capability not in the client's configured skill set.

### 7.1 Available Skill Modules (v1 — AI Data Analyst)

| Skill Module | Powered By | Description |
|---|---|---|
| SQL Querying | SQL Agent | Write and execute SQL queries against connected databases |
| Data Visualization | Chart Agent | Generate charts and graphs from query results |
| Report Generation | Orchestrator + LLM | Produce formatted text analysis and summaries |
| Scheduled Reporting | Job Queue + any agent | Autonomous recurring report generation without a human trigger |
| Hard-Rule Anomaly Detection | Anomaly Agent | Enforce client-defined business rules on every data run |
| Spreadsheet Analysis | Sheets Agent | Read from and write results to Google Sheets / Excel |
| Looker Integration | SQL Agent via Looker API | Query Looker Explores and Looks |
| Statistical Anomaly Detection | Anomaly Agent | Pattern-based deviation detection — requires 30-day baseline |
| Machine Learning | ML Agent | Forecasting (Prophet, ARIMA), classification, clustering |
| Presentation Building | Deck Agent | Auto-generate slide decks from analysis results |

### 7.2 Tier Mapping

| Skill Module | Basic | Advanced | Enterprise |
|---|---|---|---|
| SQL Querying | ✅ | ✅ | ✅ |
| Data Visualization | ✅ | ✅ | ✅ |
| Report Generation | ✅ | ✅ | ✅ |
| Scheduled Reporting | ✅ | ✅ | ✅ |
| Hard-Rule Anomaly Detection | ✅ | ✅ | ✅ |
| Spreadsheet Analysis | — | ✅ | ✅ |
| Looker Integration | — | ✅ | ✅ |
| Statistical Anomaly Detection | — | ✅ | ✅ |
| Machine Learning | — | — | ✅ |
| Presentation Building | — | — | ✅ |

### 7.3 Configuration

In v1, skill modules are configured by your team during onboarding — no self-serve UI. Changes after deployment go through a change request. Future (v2+): a self-serve skill store where clients browse and activate modules themselves.

---

## 8. Sub-agents

| Agent | Responsibility |
|---|---|
| SQL Agent | Translates requests into SQL, executes against DB, returns result set, saves successful queries as templates |
| Sheets Agent | Reads Google Sheets / Excel, handles messy non-normalized data, writes formatted output back to output tab |
| Chart Agent | Generates visualizations (bar, line, pie, scatter, heatmap) as PNG or interactive HTML. Always runs when output contains more than one data point (i.e. any trend, comparison, or multi-metric result). Single-value answers (e.g. "current NPL ratio is 3.2%") do not require a chart. |
| Deck Agent | Builds slide decks via Google Slides API or python-pptx. Uses client's branded template if provided. Async. |
| ML Agent | Trains and runs forecasting (Prophet, ARIMA), classification (churn, fraud), and clustering models. Delivers output + plain-language interpretation. Saves model artifacts per client. Async. |
| Anomaly Agent | Two modes: hard rules (day 1) + statistical baseline (day 30+). See Section 8. |

---

## 9. Anomaly Detection

### 9.1 Hard Rules — Day 1

Defined by the client during onboarding. Can be submitted as any document format (Excel table, Word doc, PDF) — parsed and extracted during knowledge base ingestion.

Example rule document format:
| Metric | Condition | Action |
|---|---|---|
| Transaction amount | > IDR 500M | Flag |
| NPL ratio | > 5% | Alert finance team |
| Column: loan_id | NULL | Error |
| Duplicate: transaction_id | Any duplicate | Flag |

Rules enforced on every data run from day one.

### 9.2 Statistical Baseline — Day 30+

After 30 days of data accumulation, the Anomaly Agent learns what "normal" looks like per metric. Then flags:
- Churn rate 3× higher than 30-day average
- Daily transactions drop 40% vs same weekday
- Revenue spike outside seasonal pattern

Both modes run concurrently after day 30.

---

## 10. Job Queue

| Task Type | Handling | ETA Communication |
|---|---|---|
| Fast (< 30s) | Synchronous — reply inline | Immediate |
| Long (minutes–hours) | Async — ACK immediately, queue job, notify on completion | "I'll have this ready in ~X minutes" |

Progress updates for very long jobs: "Still working on your churn model — 60% done, ETA 20 minutes."

On failure: AI notifies requestor with error reason and asks how to proceed.

---

## 11. Output Formats

Output format is determined by the Orchestrator from request context. Minimum for non-trivial outputs is always **text + chart**.

| Format | Trigger |
|---|---|
| Text report | All outputs — written analysis, findings, assumptions |
| Charts | All non-trivial outputs — bar, line, pie, scatter, heatmap |
| Slide deck | Request implies a presentation deliverable |
| ML model output | Forecasting / classification / clustering request |
| Data export (CSV/Excel) | Request for raw data |

---

## 12. Onboarding Checklist (per client)

1. Choose AI employee name
2. Select channels to activate (Email / Slack / WA / Jira)
3. Choose account mode: vendor-managed or client-dedicated
4. Select skill modules to enable (based on tier: Basic / Advanced / Enterprise)
5. Connect data sources (SQL credentials, Sheets, Looker API key)
6. Upload domain knowledge documents (SOPs, data dictionaries, glossaries, templates)
7. Upload hard anomaly rules document
8. Define scheduled jobs (recurring tasks, frequency, delivery destination)
9. Configure notification destinations per alert type
10. Provide branded slide template (optional, for Deck Agent)
11. Dry run: 3–5 test tasks before go-live

---

## 13. Out of Scope (MVP)

- Writing to SQL databases (read-only only)
- Looker dashboard writes (read-only for MVP)
- Self-serve client portal (your team configures everything)
- Multi-language support (Bahasa Indonesia first, English later)
- Third-party BI tools beyond Looker (Tableau, Power BI — future)
