"""
CloudLens synthetic FinOps data generator.

Design rules:
  1. FACTS ONLY. No Recommendation / EstimatedSavings / Risk / Severity columns.
     Conclusions are computed by recommend() at runtime, never stored.
  2. Daily time series: one row per resource per day.
  3. Per-service coherence: a column is populated only if it is physically
     meaningful for that service. Lambda has no CPU. EBS has no CPU. etc.
  4. Provider-native resource IDs and instance types.
  5. One seeded anomaly the agent must DISCOVER, not read off a column.
  6. Deterministic: fixed seed => identical demo every run.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)

START = pd.Timestamp("2026-04-02")
END = pd.Timestamp("2026-06-30")
DATES = pd.date_range(START, END, freq="D")

# The anomaly: a Tuesday. 12 oversized test instances launched and never killed.
SPIKE_DAY = pd.Timestamp("2026-06-16")
assert SPIKE_DAY.day_name() == "Tuesday"

HOURS = 24.0

# ---------------------------------------------------------------- price tables
# On-demand USD/hour, us-east-1 (approx, real-world magnitudes)
AWS_EC2_PRICE = {
    "m5.large": 0.096, "m5.xlarge": 0.192, "m5.2xlarge": 0.384, "m5.4xlarge": 0.768,
    "c5.large": 0.085, "c5.2xlarge": 0.340, "c5.4xlarge": 0.680,
    "r5.large": 0.126, "r5.xlarge": 0.252, "r5.2xlarge": 0.504,
    "t3.medium": 0.0416,
}
AZ_VM_PRICE = {
    "Standard_B2s": 0.0416, "Standard_D2s_v5": 0.096, "Standard_D4s_v5": 0.192,
    "Standard_D8s_v5": 0.384, "Standard_E4s_v5": 0.252, "Standard_F8s_v2": 0.338,
}
EBS_GB_MONTH = 0.08          # gp3
BLOB_GB_MONTH_HOT = 0.0184   # hot tier

AWS_REGIONS = ["us-east-1", "us-west-2", "eu-west-1"]
AZ_REGIONS = ["eastus", "westeurope", "centralindia"]
ENVS = ["prod", "staging", "dev", "test"]
BUS = ["Streaming", "Broadband", "Platform", "Analytics", "Security"]
APPS = ["vod-transcoder", "epg-service", "subscriber-api", "billing-etl",
        "cdn-origin", "recommendation-engine", "auth-gateway"]
OWNERS = ["platform-team", "cloudops", "data-eng", "security-team", "media-team"]

SUB_ID = "8f2a1c4e-9b13-4d7a-a6c5-2e0f1b7d3a99"


def hexid(n):
    return "".join(rng.choice(list("0123456789abcdef"), n))


def noise(scale=0.04):
    return float(rng.normal(1.0, scale))


def tags():
    return dict(
        environment=str(rng.choice(ENVS, p=[0.45, 0.2, 0.2, 0.15])),
        business_unit=str(rng.choice(BUS)),
        application=str(rng.choice(APPS)),
        owner=str(rng.choice(OWNERS)),
    )


COLUMNS = [
    "date", "provider", "account_id", "account_name", "region", "service",
    "resource_id", "resource_name", "instance_type",
    "environment", "business_unit", "application", "owner",
    "status", "cost_usd",
    "cpu_utilization_p95", "invocations", "attached_to",
    "storage_gb", "last_access_days",
]


def blank(**kw):
    row = {c: None for c in COLUMNS}
    row.update(kw)
    return row


# =============================================================== AWS resources
def build_aws():
    rows = []
    accounts = [("112233445566", "prod-media"), ("223344556677", "shared-services"),
                ("334455667788", "analytics")]

    # ---- EC2 baseline fleet (~50 instances, healthy mix)
    fleet = (["m5.2xlarge"] * 10 + ["m5.xlarge"] * 12 + ["c5.4xlarge"] * 8 +
             ["r5.2xlarge"] * 6 + ["m5.4xlarge"] * 8 + ["m5.large"] * 6)
    for itype in fleet:
        acct = accounts[int(rng.integers(0, 3))]
        t = tags()
        rid = f"i-0{hexid(16)}"
        name = f"{t['application']}-{itype.split('.')[0]}-{int(rng.integers(10,99))}"
        region = str(rng.choice(AWS_REGIONS, p=[0.6, 0.25, 0.15]))
        stopped = rng.random() < 0.08
        idle = (not stopped) and rng.random() < 0.06        # a few pre-existing idle
        base_cpu = rng.uniform(1.5, 4.5) if idle else rng.uniform(25, 78)
        daily = AWS_EC2_PRICE[itype] * HOURS
        for d in DATES:
            if stopped:
                # stopped EC2 accrues no compute charge, reports no CPU
                rows.append(blank(date=d, provider="AWS", account_id=acct[0],
                                  account_name=acct[1], region=region, service="EC2",
                                  resource_id=rid, resource_name=name, instance_type=itype,
                                  status="stopped", cost_usd=0.0, **t))
            else:
                rows.append(blank(date=d, provider="AWS", account_id=acct[0],
                                  account_name=acct[1], region=region, service="EC2",
                                  resource_id=rid, resource_name=name, instance_type=itype,
                                  status="running", cost_usd=round(daily * noise(), 4),
                                  cpu_utilization_p95=round(
                                      min(99.0, max(0.3, base_cpu * noise(0.12))), 1),
                                  **t))

    # ---- THE SEEDED ANOMALY: 12 x m5.4xlarge, env=test, launched SPIKE_DAY, idle forever
    daily_rogue = AWS_EC2_PRICE["m5.4xlarge"] * HOURS
    for n in range(12):
        rid = f"i-0{hexid(16)}"
        rows += [blank(date=d, provider="AWS", account_id="223344556677",
                       account_name="shared-services", region="us-east-1", service="EC2",
                       resource_id=rid, resource_name=f"loadtest-worker-{n:02d}",
                       instance_type="m5.4xlarge", environment="test",
                       business_unit="Platform", application="vod-transcoder",
                       owner="platform-team", status="running",
                       cost_usd=round(daily_rogue * noise(0.02), 4),
                       cpu_utilization_p95=round(rng.uniform(0.8, 4.2), 1))
                 for d in DATES[DATES >= SPIKE_DAY]]

    # ---- EBS volumes (no CPU; waste signal = unattached)
    ec2_ids = [r["resource_id"] for r in rows if r["service"] == "EC2"][:60]
    for i in range(40):
        acct = accounts[int(rng.integers(0, 3))]
        t = tags()
        gb = int(rng.choice([50, 100, 200, 500, 1000]))
        rid = f"vol-0{hexid(16)}"
        unattached = rng.random() < 0.15                     # 6-ish orphaned volumes
        att = None if unattached else str(rng.choice(ec2_ids))
        daily = gb * EBS_GB_MONTH / 30.0
        rows += [blank(date=d, provider="AWS", account_id=acct[0], account_name=acct[1],
                       region=str(rng.choice(AWS_REGIONS)), service="EBS",
                       resource_id=rid, resource_name=f"{t['application']}-vol-{i:02d}",
                       status="available" if unattached else "in-use",
                       cost_usd=round(daily * noise(0.01), 4),
                       attached_to=att, storage_gb=gb, **t)
                 for d in DATES]

    # ---- Lambda (no CPU, no instance type; waste signal = zero invocations)
    for i in range(15):
        acct = accounts[int(rng.integers(0, 3))]
        t = tags()
        region = str(rng.choice(AWS_REGIONS))
        fn = f"{t['application']}-fn-{i:02d}"
        rid = f"arn:aws:lambda:{region}:{acct[0]}:function:{fn}"
        dead = rng.random() < 0.27                            # 4-ish dead functions
        base_inv = 0 if dead else int(rng.integers(5_000, 400_000))
        for d in DATES:
            inv = 0 if dead else max(0, int(base_inv * noise(0.18)))
            cost = 0.0 if dead else round(inv * 2.1e-6 * noise(0.05), 4)
            rows.append(blank(date=d, provider="AWS", account_id=acct[0],
                              account_name=acct[1], region=region, service="Lambda",
                              resource_id=rid, resource_name=fn, status="active",
                              cost_usd=cost, invocations=inv, **t))
    return pd.DataFrame(rows, columns=COLUMNS)


# ============================================================= Azure resources
def build_azure():
    rows = []
    rgs = ["rg-media-prod", "rg-platform-shared", "rg-analytics"]

    # ---- Virtual Machines
    fleet = (["Standard_D4s_v5"] * 12 + ["Standard_D2s_v5"] * 10 +
             ["Standard_D8s_v5"] * 6 + ["Standard_E4s_v5"] * 6 +
             ["Standard_F8s_v2"] * 4 + ["Standard_B2s"] * 4)
    for itype in fleet:
        t = tags()
        rg = str(rng.choice(rgs))
        name = f"{t['application']}-vm-{int(rng.integers(10,99))}"
        rid = (f"/subscriptions/{SUB_ID}/resourceGroups/{rg}"
               f"/providers/Microsoft.Compute/virtualMachines/{name}")
        region = str(rng.choice(AZ_REGIONS, p=[0.5, 0.3, 0.2]))
        deallocated = rng.random() < 0.08
        idle = (not deallocated) and rng.random() < 0.10
        base_cpu = rng.uniform(1.0, 4.8) if idle else rng.uniform(22, 75)
        daily = AZ_VM_PRICE[itype] * HOURS
        for d in DATES:
            if deallocated:
                rows.append(blank(date=d, provider="Azure", account_id=SUB_ID,
                                  account_name=rg, region=region, service="Virtual Machine",
                                  resource_id=rid, resource_name=name, instance_type=itype,
                                  status="deallocated", cost_usd=0.0, **t))
            else:
                rows.append(blank(date=d, provider="Azure", account_id=SUB_ID,
                                  account_name=rg, region=region, service="Virtual Machine",
                                  resource_id=rid, resource_name=name, instance_type=itype,
                                  status="running", cost_usd=round(daily * noise(), 4),
                                  cpu_utilization_p95=round(
                                      min(99.0, max(0.3, base_cpu * noise(0.12))), 1), **t))

    # ---- Azure Functions
    for i in range(15):
        t = tags()
        rg = str(rng.choice(rgs))
        name = f"{t['application']}-func-{i:02d}"
        rid = (f"/subscriptions/{SUB_ID}/resourceGroups/{rg}"
               f"/providers/Microsoft.Web/sites/{name}")
        region = str(rng.choice(AZ_REGIONS))
        dead = rng.random() < 0.22
        base_inv = 0 if dead else int(rng.integers(3_000, 250_000))
        for d in DATES:
            inv = 0 if dead else max(0, int(base_inv * noise(0.2)))
            cost = 0.0 if dead else round(inv * 2.0e-6 * noise(0.05), 4)
            rows.append(blank(date=d, provider="Azure", account_id=SUB_ID, account_name=rg,
                              region=region, service="Azure Functions", resource_id=rid,
                              resource_name=name, status="active", cost_usd=cost,
                              invocations=inv, **t))

    # ---- Blob Storage (no CPU; waste signal = cold data on hot tier)
    for i in range(20):
        t = tags()
        rg = str(rng.choice(rgs))
        name = f"st{t['application'].replace('-','')[:12]}{i:02d}"
        rid = (f"/subscriptions/{SUB_ID}/resourceGroups/{rg}"
               f"/providers/Microsoft.Storage/storageAccounts/{name}")
        region = str(rng.choice(AZ_REGIONS))
        gb0 = float(rng.integers(200, 4000))
        growth = rng.uniform(0.0, 3.5)                        # GB/day
        cold = rng.random() < 0.25                            # 5-ish cold containers
        last_access0 = int(rng.integers(95, 400)) if cold else int(rng.integers(0, 20))
        for k, d in enumerate(DATES):
            gb = gb0 + growth * k
            rows.append(blank(date=d, provider="Azure", account_id=SUB_ID, account_name=rg,
                              region=region, service="Blob Storage", resource_id=rid,
                              resource_name=name, status="available",
                              cost_usd=round(gb * BLOB_GB_MONTH_HOT / 30.0 * noise(0.01), 4),
                              storage_gb=round(gb, 1),
                              last_access_days=last_access0 + k if cold else last_access0,
                              **t))
    return pd.DataFrame(rows, columns=COLUMNS)


if __name__ == "__main__":
    out = Path("/mnt/user-data/outputs")
    out.mkdir(parents=True, exist_ok=True)

    aws, az = build_aws(), build_azure()
    combined = pd.concat([aws, az], ignore_index=True).sort_values(
        ["date", "provider", "service", "resource_id"]).reset_index(drop=True)

    for df, stem in [(aws, "aws_finops"), (az, "azure_finops"), (combined, "finops_combined")]:
        df.to_csv(out / f"{stem}.csv", index=False)
    aws.to_excel(out / "aws_finops_data.xlsx", index=False)
    az.to_excel(out / "azure_finops_data.xlsx", index=False)

    # ---- sanity report
    ec2 = aws[aws.service == "EC2"]
    daily = ec2.groupby("date").cost_usd.sum()
    before = daily[daily.index < SPIKE_DAY].tail(14).mean()
    after = daily[daily.index >= SPIKE_DAY].mean()
    print(f"rows: aws={len(aws)} azure={len(az)} combined={len(combined)}")
    print(f"EC2 daily before spike: ${before:,.2f}")
    print(f"EC2 daily after  spike: ${after:,.2f}")
    print(f"spike magnitude: +{(after/before - 1)*100:.1f}%  on {SPIKE_DAY.date()} "
          f"({SPIKE_DAY.day_name()})")
    latest = combined[combined.date == combined.date.max()]
    print("\nwaste signals on latest day:")
    print("  idle EC2/VM (running, p95 cpu < 5):",
          len(latest[(latest.status == "running") & (latest.cpu_utilization_p95 < 5)]))
    print("  unattached EBS volumes:",
          len(latest[(latest.service == "EBS") & (latest.attached_to.isna())]))
    print("  zero-invocation functions:",
          len(latest[latest.service.isin(["Lambda", "Azure Functions"]) & (latest.invocations == 0)]))
    print("  cold blob (last access > 90d):",
          len(latest[(latest.service == "Blob Storage") & (latest.last_access_days > 90)]))
