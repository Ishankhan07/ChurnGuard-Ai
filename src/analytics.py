
import pandas as pd
from src.data_processing import load_data


def get_data():
    df = load_data().copy()

    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Churn"] = df["Churn"].astype(str).str.strip()

    return df


def churn_summary():
    df = get_data()

    total = len(df)
    churned = (df["Churn"] == "Yes").sum()
    rate = churned / total * 100 if total else 0

    return total, int(churned), rate


def churn_by_contract():
    df = get_data()

    result = df[df["Churn"] == "Yes"].groupby("Contract").size()

    return result.sort_values(ascending=False)


def churn_by_tenure():
    df = get_data()

    df["Tenure Group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 36, 48, 60, 72],
        labels=["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
    )

    result = df.groupby("Tenure Group", observed=False)["Churn"].apply(
        lambda x: (x == "Yes").mean() * 100
    )

    return result


def monthly_charges():
    df = get_data()

    result = df.groupby("Churn")["MonthlyCharges"].mean()

    return result.rename(
        index={"No": "Retained", "Yes": "Churned"}
    )


# 1. Line chart data
def churn_rate_by_tenure():
    return churn_by_tenure()


# 2. Scatter plot data
def charges_scatter_data():
    df = get_data()

    df = df.dropna(
        subset=["MonthlyCharges", "TotalCharges", "Churn"]
    )

    if len(df) > 1500:
        df = df.sample(1500, random_state=42)

    return df[["MonthlyCharges", "TotalCharges", "Churn"]]


# 3. Heatmap data
def contract_tenure_heatmap():
    df = get_data()

    df["Tenure Group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 36, 48, 60, 72],
        labels=["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
    )

    result = df.pivot_table(
        index="Contract",
        columns="Tenure Group",
        values="Churn",
        aggfunc=lambda x: (x == "Yes").mean() * 100,
        observed=False
    )

    return result


# 4. Donut chart data
def customer_churn_share():
    df = get_data()

    retained = (df["Churn"] == "No").sum()
    churned = (df["Churn"] == "Yes").sum()

    return pd.Series({
        "Retained": int(retained),
        "Churned": int(churned)
    })
