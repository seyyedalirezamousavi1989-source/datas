import numpy as np, pandas as pd
from local_chat import aurora_trade_patterns as tp, aurora_trades_v3 as v3
from local_chat.test_edge_lab import synth


def test_random_walk_has_no_medium_patterns():
    d = synth(n=40000, seed=9)
    d["epoch"] = d["epoch"] + int(pd.Timestamp("2021-09-01").timestamp()) - 1_500_000_000
    T = tp.trades_table(d)
    P = tp.mine(T)
    assert (P["tier"] == "medium").sum() == 0


def test_weak_subpattern_does_not_lift_tier():
    P = pd.DataFrame([
        dict(strategy="hour_long_21", side="long", cond=(), n_tr=600, mean_tr=0.05, win_tr=.44, p_tr=.2, n_va=200, mean_va=.1, n_te=0, mean_te=np.nan, win_te=np.nan, tier="none", n_tested=2),
        dict(strategy="hour_long_21", side="long", cond=(("dow", "Tue"),), n_tr=100, mean_tr=0.3, win_tr=.5, p_tr=.001, n_va=40, mean_va=.5, n_te=0, mean_te=np.nan, win_te=np.nan, tier="weak", n_tested=2)])
    row = {"strategy": "hour_long_21", "side": 1, "dow": "Tue"}
    parts, tier, base = v3.evidence_block(P, row)
    assert tier == "none" and base == "none" and any("فرعی" in p for p in parts)


def test_why_answer_mentions_hindsight():
    assert "آینده" in v3.WHY and "آموزشی" in v3.WHY
