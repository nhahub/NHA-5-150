import numpy as np
import pandas as pd

PAIR = ["product_name", "store_id"]
WINDOW = 7
RATIO_CAP = 10.0


def add_sales_velocity(data):
    data = data.sort_values(PAIR + ["transaction_date", "record_id"]).reset_index(drop=True)
    data["batch_rate"] = data["units_sold"] / data["days_remaining_at_purchase"].clip(lower=1)

    roller = data.groupby(PAIR).rolling(f"{WINDOW}D", on="transaction_date", closed="left")
    sold = roller["units_sold"].sum()
    rate_sum = roller["batch_rate"].sum()
    batches = roller["units_sold"].count()

    assert (sold.index.get_level_values("transaction_date") == data["transaction_date"].values).all()
    data["sold_last_7d"] = sold.fillna(0).values
    data["batches_in_window_7d"] = batches.fillna(0).astype(int).values
    window_rate = (rate_sum / batches).values

    first_date = data.groupby(PAIR)["transaction_date"].transform("min")
    data["pair_history_days"] = (data["transaction_date"] - first_date).dt.days

    daily = (data.groupby(["category", "store_id", "transaction_date"])
                 .agg(units=("units_sold", "sum"), rates=("batch_rate", "sum"), n=("batch_rate", "size")).reset_index())
    daily = daily.sort_values(["category", "store_id", "transaction_date"])
    by_group = daily.groupby(["category", "store_id"])
    units_before = by_group["units"].cumsum() - daily["units"]
    rates_before = by_group["rates"].cumsum() - daily["rates"]
    batches_before = by_group["n"].cumsum() - daily["n"]
    history_days = (daily["transaction_date"] - by_group["transaction_date"].transform("min")).dt.days
    daily["cat_store_velocity"] = units_before / history_days.clip(lower=7)
    daily["cat_store_rate"] = rates_before / batches_before.replace(0, np.nan)
    daily.loc[history_days == 0, ["cat_store_velocity", "cat_store_rate"]] = np.nan

    n_before = len(data)
    data = data.merge(daily[["category", "store_id", "transaction_date", "cat_store_velocity", "cat_store_rate"]],
                      on=["category", "store_id", "transaction_date"], how="left")
    assert len(data) == n_before

    use_own = (data["pair_history_days"] >= WINDOW) & (data["batches_in_window_7d"] > 0)
    data["sales_velocity"] = np.where(use_own, data["sold_last_7d"] / WINDOW, data["cat_store_velocity"])
    data["sales_velocity_per_day"] = np.where(use_own, window_rate, data["cat_store_rate"])
    data["velocity_source"] = np.where(use_own, "product_store_7d",
                              np.where(data["cat_store_velocity"].notna(), "category_store_avg", "no_history"))
    return data


def sell_through(data, velocity_col):
    v = data[velocity_col]
    days_left = data["days_until_expiry"].clip(lower=1)
    days_to_sell = data["current_stock"] / v.where(v > 0)
    ratio = days_to_sell / days_left
    status = np.select(
        [v.isna(), v == 0, ratio > 1],
        ["no_history", "at_risk_zero_velocity", "at_risk"], default="ok")
    ratio = ratio.where(v != 0, RATIO_CAP).clip(upper=RATIO_CAP)
    at_risk = pd.Series(np.where(v.isna(), np.nan, np.isin(status, ["at_risk", "at_risk_zero_velocity"]).astype(float)), index=data.index)
    return days_to_sell, ratio, at_risk, status
