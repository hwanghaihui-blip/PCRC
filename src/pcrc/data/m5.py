"""M5 data download and semi-synthetic environment utilities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from pcrc.constants import RAW_DATA_DIR


M5_REPO_ID = "denephew/M5_Forecasting"
M5_FILES = [
    "calendar.csv",
    "sell_prices.csv",
    "sales_train_validation.csv",
    "sales_train_evaluation.csv",
]


def _augment_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["event_name_1"] = enriched.get("event_name_1", "").fillna("none")
    enriched["event_type_1"] = enriched.get("event_type_1", "").fillna("none")
    enriched["event_name_2"] = enriched.get("event_name_2", "").fillna("none")
    enriched["event_type_2"] = enriched.get("event_type_2", "").fillna("none")
    enriched["event_flag"] = (
        (enriched["event_name_1"] != "none")
        | (enriched["event_type_1"] != "none")
        | (enriched["event_name_2"] != "none")
        | (enriched["event_type_2"] != "none")
    ).astype(int)
    enriched["event_type_primary"] = np.where(enriched["event_type_1"] != "none", enriched["event_type_1"], enriched["event_type_2"])
    return enriched


def download_m5_dataset(token: str | None = None) -> dict[str, Path]:
    target_dir = RAW_DATA_DIR / "m5"
    target_dir.mkdir(parents=True, exist_ok=True)
    resolved = {}
    for filename in M5_FILES:
        existing = target_dir / filename
        if existing.exists():
            resolved[filename] = existing
            continue
        path = hf_hub_download(
            repo_id=M5_REPO_ID,
            filename=filename,
            repo_type="dataset",
            token=token,
            local_dir=target_dir,
        )
        resolved[filename] = Path(path)
    return resolved


def build_m5_panel(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    calendar = pd.read_csv(root / "calendar.csv")
    prices = pd.read_csv(root / "sell_prices.csv")
    sales = pd.read_csv(root / "sales_train_validation.csv")
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    value_cols = [col for col in sales.columns if col.startswith("d_")]
    melted = sales.melt(id_vars=id_cols, value_vars=value_cols, var_name="d", value_name="sales")
    merged = melted.merge(calendar, on="d", how="left")
    merged = merged.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    merged["sales_lag_1"] = merged.groupby(["store_id", "item_id"])["sales"].shift(1)
    merged["sales_lag_7"] = merged.groupby(["store_id", "item_id"])["sales"].shift(7)
    merged["rolling_mean_7"] = (
        merged.groupby(["store_id", "item_id"])["sales"].rolling(window=7, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    )
    merged["rolling_mean_28"] = (
        merged.groupby(["store_id", "item_id"])["sales"].rolling(window=28, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    )
    merged["sell_price"] = merged["sell_price"].ffill().bfill()
    merged = _augment_calendar_features(merged)
    merged = merged.dropna(subset=["sell_price", "sales_lag_1", "sales_lag_7"])
    return merged


def build_m5_subset_panel(
    root: str | Path,
    states: list[str] | None = None,
    categories: list[str] | None = None,
    stores: list[str] | None = None,
    max_rows: int | None = 2000,
) -> pd.DataFrame:
    root = Path(root)
    calendar = pd.read_csv(root / "calendar.csv")
    prices = pd.read_csv(root / "sell_prices.csv")
    sales = pd.read_csv(root / "sales_train_validation.csv")
    if states is not None:
        sales = sales[sales["state_id"].isin(states)]
    if categories is not None:
        sales = sales[sales["cat_id"].isin(categories)]
    if stores is not None:
        sales = sales[sales["store_id"].isin(stores)]
    if max_rows is not None and len(sales) > max_rows:
        ranked = (
            sales.assign(series_total=sales.filter(like="d_").sum(axis=1))
            .sort_values("series_total", ascending=False)
            .head(max_rows)
            .drop(columns=["series_total"])
        )
        sales = ranked
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    value_cols = [col for col in sales.columns if col.startswith("d_")]
    melted = sales.melt(id_vars=id_cols, value_vars=value_cols, var_name="d", value_name="sales")
    merged = melted.merge(calendar, on="d", how="left")
    merged = merged.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    merged["sales_lag_1"] = merged.groupby(["store_id", "item_id"])["sales"].shift(1)
    merged["sales_lag_7"] = merged.groupby(["store_id", "item_id"])["sales"].shift(7)
    merged["rolling_mean_7"] = (
        merged.groupby(["store_id", "item_id"])["sales"].rolling(window=7, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    )
    merged["rolling_mean_28"] = (
        merged.groupby(["store_id", "item_id"])["sales"].rolling(window=28, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    )
    merged["sell_price"] = merged["sell_price"].ffill().bfill()
    merged = _augment_calendar_features(merged)
    merged = merged.dropna(subset=["sell_price", "sales_lag_1", "sales_lag_7"])
    return merged


def build_representative_subset(panel: pd.DataFrame, top_states: int = 2, top_categories: int = 2, top_stores: int = 4) -> pd.DataFrame:
    state_keep = panel["state_id"].value_counts().index[:top_states]
    cat_keep = panel["cat_id"].value_counts().index[:top_categories]
    store_keep = panel["store_id"].value_counts().index[:top_stores]
    subset = panel[
        panel["state_id"].isin(state_keep)
        & panel["cat_id"].isin(cat_keep)
        & panel["store_id"].isin(store_keep)
    ].copy()
    return subset


def build_subset_manifest(panel: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        panel.groupby(["state_id", "cat_id", "store_id"], as_index=False)
        .agg(
            n_rows=("sales", "size"),
            n_items=("item_id", "nunique"),
            min_date=("date", "min"),
            max_date=("date", "max"),
            mean_sales=("sales", "mean"),
            mean_price=("sell_price", "mean"),
        )
        .sort_values(["state_id", "cat_id", "store_id"])
    )
    return grouped


@dataclass
class M5ResponseKernel:
    elasticity_clip: tuple[float, float] = (-2.0, -0.05)
    beta_price: float = -1.0
    variance_scale: float = 0.35
    stockout_penalty: float = 0.8
    holding_cost: float = 0.05

    def deterministic_rng(self, base_seed: int, *parts: object) -> np.random.Generator:
        payload = "|".join(str(part) for part in (base_seed, *parts)).encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        seed = int.from_bytes(digest, "little") % (2**32 - 1)
        return np.random.default_rng(seed)

    @staticmethod
    def _estimate_group_elasticity(frame: pd.DataFrame, group_cols: list[str], default_beta: float) -> pd.DataFrame:
        rows = []
        for keys, group in frame.groupby(group_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            n_obs = len(group)
            unique_prices = int(group["log_price"].nunique())
            if unique_prices > 1:
                slope = float(np.polyfit(group["log_price"], group["log_sales"], deg=1)[0])
            else:
                slope = float(default_beta)
            rows.append(dict(zip(group_cols, keys)) | {"elasticity_raw": slope, "n_obs": n_obs, "unique_prices": unique_prices})
        return pd.DataFrame(rows)

    def fit_elasticity(self, panel: pd.DataFrame) -> pd.DataFrame:
        frame = panel.copy()
        frame["log_sales"] = np.log1p(frame["sales"])
        frame["log_price"] = np.log(frame["sell_price"].clip(lower=1e-3))
        item_store = self._estimate_group_elasticity(frame, ["item_id", "store_id", "cat_id", "state_id"], self.beta_price)
        store_cat = self._estimate_group_elasticity(frame, ["store_id", "cat_id", "state_id"], self.beta_price).rename(
            columns={"elasticity_raw": "elasticity_store_cat", "n_obs": "n_obs_store_cat", "unique_prices": "unique_prices_store_cat"}
        )
        state_cat = self._estimate_group_elasticity(frame, ["state_id", "cat_id"], self.beta_price).rename(
            columns={"elasticity_raw": "elasticity_state_cat", "n_obs": "n_obs_state_cat", "unique_prices": "unique_prices_state_cat"}
        )
        merged = item_store.merge(store_cat, on=["store_id", "cat_id", "state_id"], how="left")
        merged = merged.merge(state_cat, on=["state_id", "cat_id"], how="left")
        merged["elasticity_store_cat"] = merged["elasticity_store_cat"].fillna(self.beta_price)
        merged["elasticity_state_cat"] = merged["elasticity_state_cat"].fillna(self.beta_price)
        weight_item = merged["n_obs"] / (merged["n_obs"] + 30.0)
        weight_store = merged["n_obs_store_cat"].fillna(0.0) / (merged["n_obs_store_cat"].fillna(0.0) + 60.0)
        merged["elasticity"] = (
            weight_item * merged["elasticity_raw"]
            + (1.0 - weight_item)
            * (weight_store * merged["elasticity_store_cat"] + (1.0 - weight_store) * merged["elasticity_state_cat"])
        )
        merged["elasticity"] = merged["elasticity"].clip(*self.elasticity_clip)
        try:
            merged["elasticity_group"] = pd.qcut(merged["elasticity"], q=3, labels=["low", "mid", "high"], duplicates="drop")
        except ValueError:
            ranks = merged["elasticity"].rank(method="average", pct=True)
            merged["elasticity_group"] = pd.cut(
                ranks,
                bins=[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
                labels=["low", "mid", "high"],
                include_lowest=True,
            )
        return merged[["item_id", "store_id", "cat_id", "state_id", "elasticity", "elasticity_group", "n_obs", "unique_prices"]]

    def annotate_anchor_loss(self, panel: pd.DataFrame) -> pd.DataFrame:
        frame = panel.copy()
        stock_proxy = frame["rolling_mean_7"].clip(lower=1.0)
        shortfall = (frame["sales"] - stock_proxy).clip(lower=0.0)
        anchor_profit = frame["sell_price"] * np.minimum(frame["sales"], stock_proxy) - self.holding_cost * stock_proxy - self.stockout_penalty * shortfall
        frame["anchor_profit"] = anchor_profit
        frame["anchor_loss"] = -anchor_profit
        frame["promo_flag"] = ((frame.get("snap_CA", 0) + frame.get("snap_TX", 0) + frame.get("snap_WI", 0)) > 0).astype(int)
        frame["event_flag"] = frame.get("event_flag", 0).astype(int)
        frame["event_type_primary"] = frame.get("event_type_primary", "none")
        return frame

    def select_representative_series(self, panel: pd.DataFrame) -> dict[str, tuple[str, str]]:
        grouped = (
            panel.groupby(["item_id", "store_id"], as_index=False)
            .agg(elasticity=("elasticity", "mean"), n_rows=("sales", "size"))
            .query("n_rows >= 56")
        )
        if grouped.empty:
            return {"high": ("", ""), "low": ("", "")}
        high = grouped.sort_values(["elasticity", "n_rows"]).iloc[0]
        low = grouped.sort_values(["elasticity", "n_rows"], ascending=[False, False]).iloc[0]
        return {
            "high": (str(high["item_id"]), str(high["store_id"])),
            "low": (str(low["item_id"]), str(low["store_id"])),
        }

    def simulate_demand(self, row: pd.Series, price_multiplier: float, elasticity: float, rng: np.random.Generator) -> dict[str, float]:
        batch = self.simulate_demand_batch(row, price_multiplier, elasticity, rng, n_draws=1)
        return {
            "demand_mean": float(batch["demand_mean"][0]),
            "demand": float(batch["demand"][0]),
            "profit": float(batch["profit"][0]),
            "loss": float(batch["loss"][0]),
            "shortfall": float(batch["shortfall"][0]),
            "stock_proxy": float(batch["stock_proxy"][0]),
            "price": float(batch["price"][0]),
        }

    def simulate_demand_batch(
        self,
        row: pd.Series,
        price_multiplier: float,
        elasticity: float,
        rng: np.random.Generator,
        n_draws: int,
    ) -> dict[str, np.ndarray]:
        p0 = float(row["sell_price"])
        price = p0 * float(price_multiplier)
        base_mean = max(0.1, float(row["rolling_mean_28"]))
        demand_mean = base_mean * (price / max(p0, 1e-3)) ** float(elasticity)
        event_multiplier = 1.0 + 0.08 * float(row.get("event_flag", 0))
        promo_multiplier = 1.0 + 0.06 * float(row.get("promo_flag", 0))
        demand_mean = demand_mean * event_multiplier * promo_multiplier
        variance = demand_mean + self.variance_scale * demand_mean**2
        prob = np.clip(demand_mean / max(variance, 1e-6), 1e-4, 0.999)
        size = max(demand_mean * prob / max(1.0 - prob, 1e-6), 1e-3)
        draws = rng.negative_binomial(size, prob, size=int(n_draws)).astype(float)
        stock_proxy = max(1.0, float(row["rolling_mean_7"]))
        capped_sales = np.minimum(draws, stock_proxy)
        shortfall = np.clip(draws - stock_proxy, 0.0, None)
        profit = price * capped_sales - self.holding_cost * stock_proxy - self.stockout_penalty * shortfall
        return {
            "demand_mean": np.full(int(n_draws), float(demand_mean), dtype=float),
            "demand": draws,
            "profit": np.asarray(profit, dtype=float),
            "loss": np.asarray(-profit, dtype=float),
            "shortfall": np.asarray(shortfall, dtype=float),
            "stock_proxy": np.full(int(n_draws), float(stock_proxy), dtype=float),
            "price": np.full(int(n_draws), float(price), dtype=float),
        }
