"""ScreenerService._attach_market_columns 回归测试 — 大盘联动选股注入列。

覆盖:
  - 空 df 直接返回不注入
  - repo 无 get_index_daily (旧桩兼容): 指数列 null 但列存在, regime 列正常
  - 正常日: 指数 + regime 列值正确广播
  - 非交易日(regime 无当天): 取不晚于 target_date 的最近一天
  - 幂等: 列已存在不覆盖
  - _load_enriched_for_date 缓存分支注入链路
  - run() SQL 可引用大盘列筛选
  - 向后兼容: 旧策略 filter 不读大盘列, 结果不受影响
"""
from __future__ import annotations

import types
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from app.services import regime_builder
from app.services.screener import ScreenerService

TARGET = date(2026, 8, 21)  # 周五(交易日)
PREV = date(2026, 8, 20)
SUNDAY = date(2026, 8, 23)  # 周日(非交易日)


class _FakeRepo:
    """最小 repo 桩: 支持 _load_enriched_for_date 缓存分支 + get_index_daily。"""

    def __init__(self, data_dir: Path, *, enriched=None, instruments=None, index_df=None, latest=None):
        self.store = types.SimpleNamespace(data_dir=data_dir)
        self._enriched = enriched
        self._instruments = instruments if instruments is not None else pl.DataFrame()
        self._index_df = index_df if index_df is not None else pl.DataFrame()
        self._latest = latest if latest is not None else TARGET

    def get_enriched_latest_asset(self, asset_type):
        return self._enriched, self._latest

    def get_instruments_asset(self, asset_type):
        return self._instruments

    def get_enriched_history(self, target_date, lookback_days):
        return None

    def get_index_daily(self, symbol, start, end, columns=None):
        df = self._index_df
        if not df.is_empty() and "date" in df.columns:
            df = df.filter((pl.col("date") >= start) & (pl.col("date") <= end))
        if columns and not df.is_empty():
            existing = [c for c in columns if c in df.columns]
            df = df.select(existing) if existing else pl.DataFrame()
        return df


def _write_regime(data_dir: Path, rows: list[dict]) -> None:
    p = regime_builder.regime_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(p)


def _make_enriched_df() -> pl.DataFrame:
    return pl.DataFrame({
        "symbol": ["000001.SZ", "600000.SH"],
        "name": ["平安银行", "浦发银行"],
        "date": [TARGET, TARGET],
        "close": [10.0, 12.0],
        "amount": [1e8, 2e8],
        "turnover_rate": [3.0, 4.0],
        "change_pct": [0.02, 0.03],
    })


def _make_index_df() -> pl.DataFrame:
    return pl.DataFrame({
        "date": [TARGET],
        "symbol": ["000300.SH"],
        "change_pct": [0.015],
        "amplitude": [0.02],
        "amount": [3e10],
        "turnover_rate": [0.8],
    })


MARKET_COLS = (
    "index_change_pct", "index_amplitude", "index_amount", "index_turnover_rate",
    "regime_state", "regime_phase",
)


def test_empty_df_returns_empty(tmp_path):
    repo = _FakeRepo(tmp_path)
    svc = ScreenerService(repo, asset_type="stock")
    out = svc._attach_market_columns(pl.DataFrame(), TARGET)
    assert out.is_empty()


def test_repo_without_get_index_daily_fills_null_index(tmp_path):
    """旧 repo 桩无 get_index_daily: 指数列 null 但存在, regime 列正常注入。"""
    _write_regime(tmp_path, [
        {"date": TARGET, "state": "strong", "phase": "主升"},
    ])
    repo = types.SimpleNamespace(
        store=types.SimpleNamespace(data_dir=tmp_path),
    )  # 故意不带 get_index_daily
    svc = ScreenerService(repo, asset_type="stock")
    out = svc._attach_market_columns(_make_enriched_df(), TARGET)
    for col in ("index_change_pct", "index_amplitude", "index_amount", "index_turnover_rate"):
        assert col in out.columns
        assert out[col].null_count() == 2
    assert set(out["regime_state"].to_list()) == {"strong"}
    assert set(out["regime_phase"].to_list()) == {"主升"}


def test_normal_day_injects_index_and_regime(tmp_path):
    _write_regime(tmp_path, [{"date": TARGET, "state": "range", "phase": "震荡"}])
    repo = _FakeRepo(tmp_path, index_df=_make_index_df())
    svc = ScreenerService(repo, asset_type="stock")
    out = svc._attach_market_columns(_make_enriched_df(), TARGET)
    assert out["index_change_pct"].unique().to_list() == [0.015]
    assert out["index_amplitude"].unique().to_list() == [0.02]
    assert out["index_amount"].unique().to_list() == [3e10]
    assert out["index_turnover_rate"].unique().to_list() == [0.8]
    assert set(out["regime_state"].to_list()) == {"range"}
    assert set(out["regime_phase"].to_list()) == {"震荡"}
    # 广播: 两行同值
    assert out["index_change_pct"].n_unique() == 1


def test_non_trading_day_uses_latest_regime(tmp_path):
    """regime 无 SUNDAY(周日), 取不晚于它的最近一天(PREV)。"""
    _write_regime(tmp_path, [
        {"date": PREV, "state": "lean_strong", "phase": "启动"},
        {"date": TARGET, "state": "strong", "phase": "主升"},
    ])
    repo = _FakeRepo(tmp_path, index_df=pl.DataFrame({
        "date": [PREV], "symbol": ["000300.SH"],
        "change_pct": [-0.01], "amplitude": [0.015], "amount": [2e10], "turnover_rate": [0.6],
    }))
    svc = ScreenerService(repo, asset_type="stock")
    out = svc._attach_market_columns(_make_enriched_df(), SUNDAY)
    # SUNDAY 无指数数据 → null
    assert out["index_change_pct"].null_count() == 2
    # regime 取最近 TARGET(strong)
    assert set(out["regime_state"].to_list()) == {"strong"}


def test_idempotent_when_columns_exist(tmp_path):
    """列已存在不覆盖。"""
    _write_regime(tmp_path, [{"date": TARGET, "state": "weak", "phase": "冰点"}])
    repo = _FakeRepo(tmp_path, index_df=_make_index_df())
    svc = ScreenerService(repo, asset_type="stock")
    df = _make_enriched_df().with_columns(pl.lit(999.0).alias("index_change_pct"))
    out = svc._attach_market_columns(df, TARGET)
    assert out["index_change_pct"].unique().to_list() == [999.0]  # 未被覆盖


def test_load_enriched_for_date_injects_via_cache_path(tmp_path):
    """_load_enriched_for_date 命中最新缓存分支后注入大盘列。"""
    _write_regime(tmp_path, [{"date": TARGET, "state": "strong", "phase": "主升"}])
    repo = _FakeRepo(tmp_path, enriched=_make_enriched_df(), index_df=_make_index_df())
    svc = ScreenerService(repo, asset_type="stock")
    out = svc._load_enriched_for_date(TARGET)
    for col in MARKET_COLS:
        assert col in out.columns, f"missing {col}"
    assert out["index_change_pct"].n_unique() == 1
    assert set(out["regime_state"].to_list()) == {"strong"}


def test_run_sql_can_filter_by_market_column(tmp_path):
    """run() 自定义 SQL 可引用 index_change_pct 做大盘联动筛选。"""
    _write_regime(tmp_path, [{"date": TARGET, "state": "strong", "phase": "主升"}])
    repo = _FakeRepo(tmp_path, enriched=_make_enriched_df(), index_df=_make_index_df())
    svc = ScreenerService(repo, asset_type="stock")
    res = svc.run(TARGET, conditions=["index_change_pct > 0.01", "close > 11.0"], limit=10)
    # 指数涨 1.5% > 1%, 且 close>11 仅 600000.SH
    assert [r["symbol"] for r in res.rows] == ["600000.SH"]


def test_backward_compatible_legacy_filter(tmp_path):
    """旧策略 filter 不读大盘列, 结果不受注入影响。"""
    _write_regime(tmp_path, [{"date": TARGET, "state": "strong", "phase": "主升"}])
    repo = _FakeRepo(tmp_path, enriched=_make_enriched_df(), index_df=_make_index_df())
    svc = ScreenerService(repo, asset_type="stock")
    df = svc._load_enriched_for_date(TARGET)
    legacy_out = df.filter(pl.col("close") > 11.0)
    assert legacy_out["symbol"].to_list() == ["600000.SH"]
