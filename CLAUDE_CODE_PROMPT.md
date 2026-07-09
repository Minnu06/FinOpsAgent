# CloudLens — Build Spec

You are building **CloudLens**, a local multi-cloud FinOps copilot for a 24-hour hackathon.
Build it incrementally. After each phase, stop and let me verify before continuing.

---

## The one architectural rule (never violate)

**The LLM orchestrates and narrates. It never does arithmetic on raw data.**

- Python tools query the data and compute exact numbers, returning small JSON (< 30 rows).
- The LLM only chooses which tool to call and turns the returned JSON into prose.
- Never pass raw dataframes or thousands of rows into the model context.
- The system prompt must forbid stating any dollar figure that did not come from a tool result.

Consequence: every number on screen is traceable to a query, so the demo cannot hallucinate costs.

---

## Data

Files (already generated, do not regenerate):
- `data/finops_combined.csv` — 16,560 rows, 90 days (2026-04-02 → 2026-06-30)

Schema — **facts only**, one row per resource per day:

| column | notes |
|---|---|
| `date` | daily |
| `provider` | `AWS` \| `Azure` |
| `account_id`, `account_name` | AWS account / Azure subscription + RG |
| `region` | `us-east-1`, `eastus`, ... |
| `service` | `EC2`, `EBS`, `Lambda`, `Virtual Machine`, `Azure Functions`, `Blob Storage` |
| `resource_id` | provider-native (`i-0...`, `arn:aws:lambda:...`, `/subscriptions/.../virtualMachines/...`) |
| `resource_name`, `instance_type` | `instance_type` only for compute |
| `environment`, `business_unit`, `application`, `owner` | tags |
| `status` | `running`/`stopped`/`deallocated` (compute), `in-use`/`available` (EBS), `active` (functions) |
| `cost_usd` | daily cost |
| `cpu_utilization_p95` | **compute only** (null for EBS/Lambda/Blob) |
| `invocations` | **Lambda / Azure Functions only** |
| `attached_to` | **EBS only**; null = unattached |
| `storage_gb` | EBS / Blob only |
| `last_access_days` | **Blob only** |

Columns are null where physically meaningless. Never invent CPU for a disk.

**There are NO recommendation/savings/risk columns.** Recommendations are computed at runtime.

### The seeded anomaly (the agent must DISCOVER this, never read it)
On **Tuesday 2026-06-16**, twelve `m5.4xlarge` EC2 instances named `loadtest-worker-*`
(`environment=test`, account `shared-services`, region `us-east-1`) appear and run at
**1–4% CPU** for the rest of the window. AWS EC2 spend goes **$492/day → $714/day (+45%)**.

---

## Layer 1 — Adapter (build first)

`adapters/base.py` — a `CloudAdapter` Protocol:

```python
class CloudAdapter(Protocol):
    provider: str
    def get_cost(self, start: date, end: date, service: str | None = None,
                 region: str | None = None, group_by: list[str] | None = None) -> pd.DataFrame: ...
    def get_utilization(self, resource_ids: list[str]) -> pd.DataFrame: ...
    def get_metadata(self, resource_ids: list[str]) -> pd.DataFrame: ...
```

`adapters/synthetic.py` — `SyntheticAdapter(provider)` reads the CSV, filters by provider.
`adapters/multi.py` — `MultiCloudAdapter([...])` fans out and concatenates.

This is the ONLY layer that changes in v2 (an `AWSAdapter` using boto3:
`ce.get_cost_and_usage`, `cloudwatch.get_metric_data`, `ec2.describe_instances`).
Nothing above this layer may import pandas-specific or provider-specific logic.

---

## Layer 2 — Tools (deterministic, no LLM inside)

`tools/finops_tools.py`. Each returns a small dict. **No tool may return more than ~30 records.**

### 1. `cost_trend(start, end, service=None, provider=None, granularity="day")`
Returns daily totals + summary stats:
```json
{"series": [{"date": "...", "cost": 492.31}, ...],
 "total": 44308.0, "avg_daily": 492.3, "pct_change_first_last": 45.1}
```

### 2. `detect_spike(lookback_days=30, provider=None, service=None)`
Compare each day against the trailing 14-day mean. Flag days exceeding **2 standard deviations**
or **>25% jump**. For the largest spike, attribute the driver by diffing resource sets
before/after. Returns:
```json
{"spike_date": "2026-06-16", "service": "EC2", "provider": "AWS", "region": "us-east-1",
 "baseline_daily": 492.31, "spiked_daily": 714.27, "pct_increase": 45.1,
 "delta_usd_per_day": 221.96,
 "driver_resource_ids": ["i-0...", ...],
 "driver_summary": {"count": 12, "instance_type": "m5.4xlarge", "environment": "test",
                    "first_seen": "2026-06-16"}}
```
`driver_resource_ids` is deliberately shaped to feed straight into the next tool.

### 3. `find_idle_resources(provider=None, service=None, resource_ids=None)`
Four rules, one per service archetype — **each maps to one real API call in v2**:

| service | idle rule | v2 source |
|---|---|---|
| EC2 / Virtual Machine | `status == running` AND `cpu_utilization_p95 < 5` (7-day avg) | CloudWatch `CPUUtilization` |
| EBS | `attached_to` is null | `ec2.describe_volumes` |
| Lambda / Azure Functions | `invocations == 0` over the window | CloudWatch `Invocations` |
| Blob Storage | `last_access_days > 90` | Storage metrics |

Returns a list of `{resource_id, service, reason, evidence, monthly_cost_usd}`.

### 4. `recommend(resource_ids)`
Pure rules over facts. **This is a function, never a data column.**

- idle compute (`cpu_p95 < 5`)  → `terminate`, saving = full monthly cost
- underutilized compute (`5 ≤ cpu_p95 < 40`) → `downsize` one size down, saving ≈ 50%
- unattached volume → `delete_volume` (or snapshot first), saving = full monthly cost
- zero-invocation function → `decommission`, saving = full monthly cost
- cold blob (`last_access_days > 90`) → `move_to_archive_tier`, saving ≈ 80%

Returns `{recommendations: [...], total_monthly_saving_usd: N}`. Every recommendation must carry
`reason` and `evidence` fields citing the fact that triggered it.

---

## Layer 3 — Agent

`agent/loop.py`. Tool-calling loop:

```
messages = [system, user]
loop:
    resp = llm.chat(messages, tools=SCHEMAS)
    if no tool_calls: break
    for call: result = REGISTRY[call.name](**call.args); append {"role":"tool", ...}
```

`agent/llm.py` — one thin abstraction over OpenAI and Ollama, selected by env var
`LLM_PROVIDER=openai|ollama`. Default `openai` / `gpt-4o-mini` for routing reliability.
Ollama must work as a config flag (this is the "runs fully local" story).

### Tool schema descriptions
Write descriptions that state **trigger conditions and explicit negatives**, e.g.:
> `detect_spike`: "Find the day cloud cost anomalously increased and identify the resources
> driving it. Use when the user asks why cost went up or what caused a spike.
> Do NOT use for plain spend totals — use `cost_trend` for that."

### System prompt requirements
- Role: autonomous FinOps agent for AWS + Azure.
- Investigation order: find the anomaly → identify driver resources → check if they are idle → recommend.
- **Provider inference**: if the user names an AWS-only service (EC2/EBS/Lambda) filter to AWS;
  Azure-only service → Azure; if ambiguous or the user says "cloud"/"our bill", **scan BOTH**
  and report a per-provider breakdown. Never ask the user which cloud.
- **Never state a dollar figure, percentage, resource count, or resource ID that did not appear
  in a tool result.** If a tool returned nothing, say so.
- Answer format: one-paragraph root cause, then a bulleted recommendation list with savings,
  then the total monthly saving.

---

## Layer 4 — UI

`app.py` using **Chainlit**. Requirements:
- Chat interface, streaming responses.
- Each tool call rendered as a visible `cl.Step` showing tool name, arguments, and returned JSON.
  This visible chain is the proof the system is agentic — do not hide it.
- A "Scan for anomalies" starter button that injects the proactive prompt:
  `"Scan the last 30 days for cost anomalies and idle waste across all clouds and report findings."`
  This is the **proactive loop** — same agent, same tools, only the first message differs.

---

## Build order (stop after each phase)

1. `adapters/` + load CSV + a `scripts/smoke_adapter.py` printing 5 rows per method.
2. `tools/finops_tools.py` + `pytest` tests asserting:
   - `detect_spike` returns `spike_date == 2026-06-16` and exactly 12 driver resources
   - `find_idle_resources` returns >0 of each of the four archetypes
   - `recommend` returns a positive `total_monthly_saving_usd`
3. `agent/` loop + CLI (`python -m agent.cli "why did EC2 cost go up?"`). Verify the chain:
   `detect_spike → find_idle_resources → recommend`.
4. `app.py` Chainlit UI with visible steps + starter button.
5. Only then: polish, error handling, a `README.md` with the v2 boto3 swap documented.

---

## Constraints

- Python 3.11+, `uv` for env management.
- Runs entirely locally. No cloud calls in v1.
- Dependencies: `pandas`, `chainlit`, `openai`, `ollama`, `pytest`, `python-dotenv`. Nothing heavier.
- Type-hint everything. Keep tool functions pure and independently testable.
- No `Recommendation`/`Savings`/`Risk`/`Severity` columns may ever be added to the data.

Start with Phase 1. Show me the adapter code and the smoke test output before moving on.
