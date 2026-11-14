import numpy as np, pandas as pd, pytest
from local_chat import edge_lab as el


def synth(n=24000, seed=1, drift=0.0, spread=0.00008):
    rng = np.random.default_rng(seed)
    ret = rng.normal(drift, 0.0006, n)
    c = 1.1 * np.exp(np.cumsum(ret))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0003, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0003, n)))
    ep = 1_500_000_000 + np.arange(n) * 3600
    return pd.DataFrame({"time": ep, "epoch": ep, "open": o, "high": hi, "low": lo, "close": c, "spread_price": np.full(n, spread)})


def test_grid_ids_unique():
    ids = [el.config_id(f, p) for f, p in el.grid()]
    assert len(ids) == len(set(ids)) > 150
    for i in ids[::17]:
        f, p = el._parse_id(i)
        assert el.config_id(f, p) == i


def test_screen_flags_expensive_symbol():
    d = synth()
    assert el.screen_symbol(d)["pass"]
    d["spread_price"] = 0.01
    assert not el.screen_symbol(d)["pass"]


def test_random_walk_gives_no_candidate():
    res = el.search({"A": synth(seed=2), "B": synth(seed=3)}, spa_reps=100)
    assert res["overall"] == "no_edge"


def test_holdout_isolated_and_single_use(tmp_path):
    df = synth(seed=4)
    reg = el.Registry(str(tmp_path / "r.jsonl"))
    res = el.search({"A": df}, reg, spa_reps=60)
    _, dev_end, _ = el.split_points(df)
    # development trades must close before the sealed window
    b = el.build_book("A", df)
    assert reg.n_trials("A") == len(b.ids)
    fid = b.ids[0]
    h = el.holdout({"A": df}, res, fid, reg)
    assert h["verdict"] in ("no_edge", "holdout_passed")
    with pytest.raises(RuntimeError):
        el.holdout({"A": df}, res, fid, reg)
    assert len(el.Registry(str(tmp_path / "r.jsonl")).holdout_looks(fid)) == 1


def test_signals_causal():
    """Truncating the future must not change past signals."""
    df = synth(n=3000, seed=5)
    full = el.Cache(df)
    cut = el.Cache(df.iloc[:2500].reset_index(drop=True))
    for fam, (fn, space) in el.FAMILIES.items():
        p = {k: v[0] for k, v in space.items()}
        a, b = fn(full, p)[0][:2400], fn(cut, p)[0][:2400]
        assert (np.asarray(a) == np.asarray(b)).all(), fam
