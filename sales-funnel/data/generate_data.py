"""
Synthetic sales-funnel data generator.

Generates a fully synthetic Prospect -> Opportunity -> Sale dataset for a
fictional multi-dealership auto retail network ("AutoCo"), inspired by the
general structure of a real sales-funnel conversion study (marketing
campaign, brand, region, funnel stage, dates), but with entirely invented
records, dealership names, and distributions. No real business data is
used anywhere in this repository.

Run with: python data/generate_data.py
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta

RNG = np.random.default_rng(42)
N_PROSPECTS = 8000

DEALERSHIPS = ["Norte Motors", "Sur Auto", "Centro Cars", "Este Vehiculos", "Oeste Motors", "Delta Auto"]
BRANDS = ["Fiat", "Peugeot", "Renault", "Volkswagen", "Nissan", "Jeep", "Ford", "Chevrolet", "Toyota", "No Brand"]
BRAND_PROBS = np.array([0.22, 0.19, 0.15, 0.14, 0.09, 0.07, 0.05, 0.04, 0.03, 0.02])
BRAND_PROBS = BRAND_PROBS / BRAND_PROBS.sum()

REGIONS = ["Central", "North", "South", "Coast", "Mountain", "East Valley", "West Valley", "Capital"]
CAMPAIGNS = ["Dealer Website", "Brand Website", "Google - Mixed", "Google - Commercial",
             "Facebook - Mixed", "Facebook - Commercial", "Referral", "Walk-in", "WhatsApp"]
REGISTRATION_TYPE = ["New (0km)", "Used Vehicle", "Special Promotion", "Financed Plan"]
OWNER_ROLE = ["Sales Rep", "Agent", "Supervisor", "Manager"]

PROSPECT_STATUS = ["Disqualified", "Converted", "New", "Assigned", "In Progress"]
PROSPECT_STATUS_PROBS = [0.54, 0.28, 0.11, 0.04, 0.03]


def random_date(start, end):
    delta = (end - start).days
    return start + timedelta(days=int(RNG.integers(0, delta + 1)))


def simulate_prospects(n=N_PROSPECTS):
    df = pd.DataFrame({
        "prospect_id": np.arange(1, n + 1),
        "dealership": RNG.choice(DEALERSHIPS, n),
        "brand_interest": RNG.choice(BRANDS, n, p=BRAND_PROBS),
        "region": RNG.choice(REGIONS, n),
        "campaign": RNG.choice(CAMPAIGNS, n),
        "registration_type": RNG.choice(REGISTRATION_TYPE, n),
        "owner_role": RNG.choice(OWNER_ROLE, n, p=[0.55, 0.30, 0.10, 0.05]),
        "days_in_prospect_stage": RNG.gamma(shape=2.0, scale=5.0, size=n).round().astype(int).clip(0, 90),
    })

    creation = [random_date(date(2024, 1, 1), date(2025, 12, 31)) for _ in range(n)]
    df["created_date"] = creation

    # ------------------------------------------------------------------
    # Simulate conversion probability as a function of a few realistic
    # drivers, mirroring the kind of relationships found in the original
    # study (shorter time-in-stage and defined brand intent -> higher
    # conversion), then sample a binary outcome from it.
    # ------------------------------------------------------------------
    base = -1.0
    logit = (
        base
        - 0.05 * df["days_in_prospect_stage"]
        + np.where(df["brand_interest"] != "No Brand", 0.6, -0.3)
        + np.where(df["campaign"].isin(["Referral", "Walk-in", "Dealer Website"]), 0.5, 0.0)
        + np.where(df["registration_type"] == "New (0km)", 0.4, 0.0)
        + np.where(df["owner_role"].isin(["Supervisor", "Manager"]), 0.3, 0.0)
        + RNG.normal(0, 0.8, n)
    )
    prob_convert = 1 / (1 + np.exp(-logit))
    df["converted"] = (RNG.uniform(0, 1, n) < prob_convert).astype(int)

    return df


def simulate_opportunities(prospects: pd.DataFrame):
    opp = prospects[prospects["converted"] == 1].copy()
    n = len(opp)

    opp["days_in_opportunity_stage"] = RNG.gamma(shape=2.5, scale=6.0, size=n).round().astype(int).clip(0, 120)
    opp["kept_same_brand"] = RNG.choice([0, 1], n, p=[0.35, 0.65])

    logit = (
        -0.8
        - 0.04 * opp["days_in_opportunity_stage"]
        + np.where(opp["kept_same_brand"] == 1, 0.5, -0.2)
        + np.where(opp["brand_interest"].isin(["Ford", "Toyota", "No Brand"]), 0.5, 0.0)
        + np.where(opp["registration_type"] == "Used Vehicle", 0.35, 0.0)
        + np.where(opp["owner_role"].isin(["Supervisor", "Manager"]), 0.3, 0.0)
        + RNG.normal(0, 0.8, n)
    )
    prob_sale = 1 / (1 + np.exp(-logit))
    opp["sale"] = (RNG.uniform(0, 1, n) < prob_sale).astype(int)

    return opp


def generate(output_dir="data"):
    from pathlib import Path
    prospects = simulate_prospects()
    opportunities = simulate_opportunities(prospects)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prospects.to_csv(out / "prospects.csv", index=False)
    opportunities.to_csv(out / "opportunities.csv", index=False)
    return prospects, opportunities


def main():
    prospects, opportunities = generate()
    print(f"prospects.csv: {len(prospects)} rows, conversion rate = {prospects['converted'].mean():.2%}")
    print(f"opportunities.csv: {len(opportunities)} rows, sale rate = {opportunities['sale'].mean():.2%}")


if __name__ == "__main__":
    main()
