"""Chronological split with purge + embargo (decision time = bar close)."""
import numpy as np
import pandas as pd

from . import s5config as S

LABEL_END_HOURS = 24      # conservative: any label / management replay of a decision ends within 24h of entry


def assign_splits(lab: pd.DataFrame) -> pd.DataFrame:
    sp = S.SPLIT
    ts = lambda k: pd.Timestamp(sp[k])  # noqa: E731
    bc = lab["bar_close_time"]
    end = lab["entry_time"] + pd.Timedelta(hours=LABEL_END_HOURS)
    emb = pd.Timedelta(minutes=S.DECISION_TF_MIN * sp["embargo_bars"])
    split = np.full(len(lab), "", dtype=object)
    reason = np.full(len(lab), "", dtype=object)
    inT = bc <= ts("train_end")
    inV = (bc >= ts("val_start")) & (bc <= ts("val_end"))
    inX = bc >= ts("test_start")
    split[inT], split[inV], split[inX] = "train", "val", "test"
    # purge: label/replay window must finish before the next split starts
    purge_T = inT & (end >= ts("val_start"))
    purge_V = inV & (end >= ts("test_start"))
    # embargo: no decisions in the first `embargo` after a boundary
    emb_V = inV & (bc < ts("val_start") + emb)
    emb_X = inX & (bc < ts("test_start") + emb)
    for m, r in ((purge_T, "purged_before_val"), (purge_V, "purged_before_test"), (emb_V, "embargo_after_train"), (emb_X, "embargo_after_val")):
        reason[m.values] = r
    gap = ~(inT | inV | inX)
    split[gap.values], reason[gap.values] = "none", "between_splits"
    out = lab[["time", "idx"]].copy()
    out["split"] = np.where(reason != "", "excluded", split)
    out["nominal_split"] = split
    out["split_exclusion"] = reason
    return out


def audit(lab: pd.DataFrame, sp: pd.DataFrame) -> dict:
    m = lab.merge(sp[["idx", "split"]], on="idx")
    v = m[m.valid]
    rep = {"config": S.SPLIT, "label_end_hours": LABEL_END_HOURS, "splits": {}}
    for s in ("train", "val", "test"):
        x = v[v.split == s]
        rep["splits"][s] = {"rows": int(len(x)), "first_bar_close": str(x.bar_close_time.min()), "last_bar_close": str(x.bar_close_time.max()),
                            "last_label_window_end": str((x.entry_time + pd.Timedelta(hours=LABEL_END_HOURS)).max()),
                            "label_distribution": x.label.value_counts(normalize=True).round(4).to_dict()}
    tr, va, te = (v[v.split == s] for s in ("train", "val", "test"))
    checks = {
        "train_label_windows_end_before_val_start": bool((tr.entry_time + pd.Timedelta(hours=LABEL_END_HOURS)).max() < pd.Timestamp(S.SPLIT["val_start"])),
        "val_label_windows_end_before_test_start": bool((va.entry_time + pd.Timedelta(hours=LABEL_END_HOURS)).max() < pd.Timestamp(S.SPLIT["test_start"])),
        "no_time_overlap_train_val": bool(tr.bar_close_time.max() < va.bar_close_time.min()),
        "no_time_overlap_val_test": bool(va.bar_close_time.max() < te.bar_close_time.min()),
        "embargo_respected_val": bool(va.bar_close_time.min() >= pd.Timestamp(S.SPLIT["val_start"]) + pd.Timedelta(minutes=15 * S.SPLIT["embargo_bars"])),
        "embargo_respected_test": bool(te.bar_close_time.min() >= pd.Timestamp(S.SPLIT["test_start"]) + pd.Timedelta(minutes=15 * S.SPLIT["embargo_bars"])),
        "split_index_sets_disjoint": bool(len(set(tr.idx) & set(va.idx)) == 0 and len(set(va.idx) & set(te.idx)) == 0 and len(set(tr.idx) & set(te.idx)) == 0),
    }
    rep["checks"], rep["all_pass"] = checks, all(checks.values())
    rep["excluded_by_reason"] = sp.split_exclusion.value_counts().to_dict()
    return rep
