# CloudLens

A local multi-cloud FinOps copilot for AWS and Azure. Ask why a bill changed or what's
idle; the agent investigates with deterministic tools and narrates exact numbers back to
you — it never invents a dollar figure.

## The one architectural rule

**The LLM orchestrates and narrates. It never does arithmetic on raw data.**

- Python tools (`tools/finops_tools.py`) query the data and compute exact numbers,
  returning small JSON (≤ 30 records).
- The LLM only chooses which tool to call and turns the returned JSON into prose.
- The system prompt forbids stating any dollar figure, percentage, resource count, or
  resource ID that did not come from a tool result.

Every number on screen is traceable to a query — the demo cannot hallucinate costs.

The same discipline applies to dates: the system prompt is grounded with the dataset's
actual min/max available dates every turn, so the model treats the data's most recent
day as "today" instead of guessing a year from its own training cutoff. `cost_trend`'s
`start`/`end` are optional and default to the last 30 days of available data — the model
is instructed to omit them for relative questions ("last 30 days", "recently") rather
than compute a date itself.

## Architecture

```
adapters/       CloudAdapter Protocol + SyntheticAdapter (CSV) + MultiCloudAdapter (fan-out)
                + factory.py (provider -> adapter registry)
resolvers/      service_registry.py — canonical service catalog + synonym/fuzzy resolution
tools/          finops_tools.py — cost_trend, detect_spike, find_idle_resources, recommend
agent/          llm.py (OpenAI/Ollama), tool_schemas.py, loop.py (tool-calling loop), cli.py
app.py          Chainlit UI — visible cl.Step per tool call, streamed final answer
logging_setup.py  Per-session logging (logs/) shared by every layer above
```

Only `adapters/` knows how data is fetched. Everything above it works with plain
`pandas.DataFrame` results and provider-agnostic filters — this is the only layer that
changes in v2 (see [Swapping in real clouds](#swapping-in-real-clouds-v2) below).

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) for environment management
- An OpenAI API key (default backend), **or** a local [Ollama](https://ollama.com)
  install with a tool-calling model pulled (fully local backend)

## Setup

```bash
git clone git@github.com:Minnu06/FinOpsAgent.git
cd FinOpsAgent
uv sync          # creates .venv/ and installs pandas, chainlit, openai, ollama, pytest, python-dotenv
```

`data/finops_combined.csv` must be present (16,560 rows, 90 days, AWS + Azure). It is
already checked into this repo — do not regenerate it, since the demo depends on a
specific seeded anomaly (see below).

## Adding your OpenAI API key

CloudLens reads configuration from a `.env` file in the repo root (via `python-dotenv`).
It is gitignored, so your key never gets committed.

```bash
cp .env.example .env
```

Edit `.env` and set:

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...your key...
OPENAI_MODEL=gpt-4o-mini
```

Get a key at https://platform.openai.com/api-keys. `gpt-4o-mini` is the default —
it's the cheapest model with reliable tool-call routing, which this project leans on
heavily (the agent chains 2-3 tool calls per investigation).

## Running it locally

All commands run through `uv run`, which uses the project's `.venv` automatically —
no manual `source .venv/bin/activate` needed.

### 1. Smoke-test the adapter layer

```bash
uv run python -m scripts.smoke_adapter
```

Prints 5 sample rows per adapter method (AWS, Azure, and multi-cloud fan-out).

### 2. Run the test suite

```bash
uv run pytest tests/ -v
```

Covers the deterministic tools (`detect_spike` finds the seeded 2026-06-16 anomaly with
exactly 12 driver resources; `find_idle_resources` covers all four idle archetypes;
`recommend` produces a positive total saving) and the agent loop's multi-turn tool-calling
mechanics (via a scripted fake LLM client, independent of any real model's reasoning
quality).

### 3. Try the CLI

```bash
uv run python -m agent.cli "why did EC2 cost go up?"
```

Prints each tool call (name, arguments, JSON result) as it happens, then the final
narrated answer. Good for verifying the investigation chain
(`detect_spike → find_idle_resources → recommend`) without a browser. Add
`--provider AWS` or `--provider Azure` to hard-scope the query to one cloud (same
mechanism the UI dropdown uses):

```bash
uv run python -m agent.cli "why did cost go up?" --provider AWS
```

### 4. Launch the prototype (Chainlit UI)

```bash
uv run chainlit run app.py
```

Opens at **http://localhost:8000**. If that port is taken, pass `--port <n>`.

- Chat interface with streamed responses.
- **Provider dropdown**: click the settings (gear) icon to scope the whole chat to
  **AWS**, **Azure**, or **Both** (default). This is a hard filter, not a hint — every
  tool call in that session is forced to the selected provider's `provider` argument
  regardless of what the model requests (see `test_provider_override_wins_over_model_request`
  in `tests/test_agent_loop.py`).
- **Visible tool-calling flow**: each user message opens a "🔎 Investigating" step, and
  every tool call the agent makes renders as a nested step inside it — name, arguments,
  and JSON result — in the order they were called. This is the visible proof the system
  is agentic; it's never hidden.
- Click the **"Scan for anomalies"** starter to trigger the proactive investigation
  prompt (same agent, same tools — only the first message differs).

To stop the server, `Ctrl+C` in the terminal it's running in.

- **Conversation memory**: within a chat session, follow-up questions ("what about
  Azure?", "how much would that save?") have context — prior turns (including tool
  results already fetched) are threaded into the next call. A new browser tab/session
  starts fresh. The CLI is single-shot per process, so history doesn't carry between
  separate `agent.cli` invocations.

## Logs and debugging

Every session — CLI run or Chainlit chat — gets its own log file under `logs/`
(`logs/<timestamp>_<cli|chainlit>_<session_id>.log`), created by `logging_setup.py`.

- **Console** (visible while running): a clean, one-line-per-event narrative — the user
  query, which tool was called with what arguments, and the result summary. This is
  what you see in the CLI output above.
- **File** (`logs/*.log`): the same narrative at DEBUG level, plus the full JSON payload
  of every tool call/result and every adapter-level query (`SyntheticAdapter.get_cost`,
  `MultiCloudAdapter` fan-out, row counts) — enough to reconstruct exactly what happened
  in a session after the fact.

Log files are gitignored (`logs/*.log`); the folder itself is tracked via `logs/.gitkeep`
so it always exists.

## Running fully local (no OpenAI calls)

Requires [Ollama](https://ollama.com) installed and running, with a tool-calling model
pulled:

```bash
ollama pull llama3.1
```

Then in `.env`:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1
```

Or as one-off env vars without editing `.env`:

```bash
LLM_PROVIDER=ollama OLLAMA_MODEL=llama3.1 uv run python -m agent.cli "what's idle in AWS?"
```

**Known limitation:** small local models (tested with `llama3.2:3b`) reliably make the
*first* tool call correctly but sometimes stop short of chaining through
`detect_spike → find_idle_resources → recommend`, answering early with fabricated
numbers instead — despite the system prompt's hard rule against it. This is a model
capability limitation, not a bug in the agent loop (verified separately via a scripted
fake-client test, see `tests/test_agent_loop.py`). Larger local models (`llama3.1:8b`,
`qwen2.5:7b`+) are noticeably more reliable at multi-step tool chaining. `gpt-4o-mini` is
the recommended default specifically because tool-call routing reliability is the
project's main correctness risk.

## The seeded anomaly (what the agent should discover)

On **Tuesday 2026-06-16**, twelve `m5.4xlarge` EC2 instances named `loadtest-worker-*`
(`environment=test`, account `shared-services`, region `us-east-1`) appear and run at
1-4% CPU for the rest of the window. AWS EC2 spend jumps ~68% overnight. The agent must
*discover* this via `detect_spike` — it is never read from a data column.

## Swapping in real clouds (v2)

`adapters/` is the only layer that changes. Everything in `tools/`, `agent/`, and
`app.py` is written against the `CloudAdapter` Protocol
(`get_cost`, `get_utilization`, `get_metadata`, `list_services`) and never imports
pandas-specific or provider-specific logic — so a real adapter is a drop-in
replacement.

```python
# adapters/aws.py (v2, not yet implemented)
class AWSAdapter:
    provider = "AWS"

    def get_cost(self, start, end, service=None, region=None, group_by=None):
        # boto3 Cost Explorer: ce.get_cost_and_usage(...)
        ...

    def get_utilization(self, resource_ids):
        # boto3 CloudWatch: cloudwatch.get_metric_data(...) for CPUUtilization / Invocations
        ...

    def get_metadata(self, resource_ids):
        # boto3 EC2/Lambda/S3 describe calls: ec2.describe_instances(...), etc.
        ...

    def list_services(self):
        # which services this account actually has cost data for — used by the
        # validation layer to report "data not available" instead of a silent
        # empty result. A real adapter might cache this from a first Cost
        # Explorer call rather than re-querying per request.
        ...
```

Then register it in `adapters/factory.py` — the single place every other module asks
"give me the adapter for provider X":

```python
from adapters.factory import register

register("AWS", AWSAdapter())      # was register("AWS", SyntheticAdapter("AWS"))
register("Azure", AzureAdapter())  # was register("Azure", SyntheticAdapter("Azure"))
```

No changes needed to `detect_spike`, `find_idle_resources`, `recommend`, the tool
schemas, the agent loop, the resolver/validation layer, or the Chainlit UI — they only
ever see `pandas.DataFrame` results shaped by the Protocol, never the underlying API or
how adapters are constructed.

| Rule (`find_idle_resources`) | v1 (synthetic) | v2 (real) |
|---|---|---|
| EC2 / VM idle | `cpu_utilization_p95 < 5` | CloudWatch `CPUUtilization` metric |
| EBS unattached | `attached_to` is null | `ec2.describe_volumes` |
| Lambda / Functions zero-invocation | `invocations == 0` | CloudWatch `Invocations` metric |
| Blob cold | `last_access_days > 90` | Storage account access-time metrics |

## Repo & permissions

This repo enforces a branch-per-change workflow and a package-install approval gate —
see `CLAUDE.md` and `.claude/settings.json`. In short: never push to `main` directly,
and never install a new dependency without asking first.
