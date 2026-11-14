"""AURORA dataset v3: outcome-labelled setup trades -> evidence-aware decisions and post-trade reviews.

The point is NOT to teach the model which trades win.  Each row gives the model (a) a real, causally built market context,
(b) a mechanical setup with exact levels, (c) historical evidence for the setup and the matching context patterns (mined on train only,
BH-corrected, confirmed on val, see aurora_trade_patterns) and the model must answer from that evidence.  Outcomes (win/loss) appear only in
post-trade reviews, where the model learns to separate decision quality from luck.  Rows never reveal a future outcome before it is known.

    python -m local_chat.aurora_trades_v3 --out training_data/aurora_trades_v3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from typing import Dict, List

import pandas as pd

try:
    from . import aurora_forex_dataset as v1
    from . import aurora_forex_v2 as v2
    from . import aurora_trade_patterns as tp
    from . import edge_lab as el
except ImportError:                       # pragma: no cover
    import aurora_forex_dataset as v1
    import aurora_forex_v2 as v2
    import aurora_trade_patterns as tp
    import edge_lab as el

DISCLAIMER = v1.DISCLAIMER
TIER_FA = {"medium": "اعتبار متوسط", "weak": "اعتبار ضعیف", "none": "بدون شواهد کافی"}
SIDE_FA = {1: "خرید", -1: "فروش"}
ASK_EVAL = ["ستاپ «{s}» الان فعال شده. بگیرمش؟", "سیگنال {s} روی {sym} آمده؛ ارزش ورود داره؟", "{s} روی {sym} فعال شد، با توجه به شواهد چه تصمیمی بگیرم؟",
            "ستاپ {s} رو ارزیابی کن: بگیرم یا ردش کنم؟", "الان سیگنال {s} داریم. سطوح و شواهد رو بگو."]
ASK_REVIEW = ["این معامله ({s}) بسته شد: {o}. تصمیمم درست بود؟", "نتیجهٔ معامله‌ی {s}: {o}. ارزیابی کن که ورودم اشتباه بود یا نه.", "معامله {s} با {o} بسته شد. چی یاد بگیرم؟"]
ASK_TABLE = ["کدوم الگوها شواهد بهتری دارن؟", "تا الان کدوم شرایط معاملاتی اعتبار متوسط یا ضعیف دارن؟", "جدول الگوهای تاریخی رو خلاصه کن؛ چیزی هست که ارزش دنبال کردن داشته باشه؟"]
ASK_WHY = ["چرا فقط معاملات برنده رو انتخاب نکنیم و مدل رو روی اونا آموزش ندیم؟", "اگه فقط معاملات درست رو به مدل نشون بدیم، یاد می‌گیره سود کنه؟",
           "می‌شه از روی معاملات برندهٔ گذشته الگو پیدا کرد و مدل رو آموزش داد؟"]


def fmt_r(x):
    return f"{x:+.2f}R"


def cond_text(cond):
    return " + ".join(f"{v}" if k in ("hblk", "dow") else VAL(v) for k, v in cond) if cond else "بدون شرط اضافه"


def VAL(v):
    return tp.VAL_FA.get(v, v)


def pat_line(r):
    c = cond_text(r["cond"])
    va = "-" if pd.isna(r["mean_va"]) or r["n_va"] == 0 else f"{r['mean_va']:+.2f}R (n={int(r['n_va'])})"
    return f"{TIER_FA[r['tier']]} | {tp.FA[r['strategy']]} | {c} | آموزش: n={int(r['n_tr'])}، میانگین {fmt_r(r['mean_tr'])}، برد {r['win_tr'] * 100:.0f}% | تأیید: {va}"


def matches(P: pd.DataFrame, strat, side, feats):
    sn = "long" if side == 1 else "short"
    m = P[(P["strategy"] == strat) & (P["side"] == sn)]
    ok = m[pd.Series([all(feats.get(k) == v for k, v in c) for c in m["cond"]], index=m.index, dtype=bool)]
    ok = ok.assign(_o=ok["tier"].map({"medium": 0, "weak": 1, "none": 2}), _c=[len(c) for c in ok["cond"]])
    return ok.sort_values(["_o", "_c", "p_tr"], ascending=[True, False, True])


def evidence_block(P, row, digits=2):
    m = matches(P, row["strategy"], row["side"], row)
    base = m[m["cond"].apply(len) == 0]
    parts = []
    base_tier = "none"
    if len(base):
        b = base.iloc[0]
        base_tier = b["tier"]
        parts.append("خانوادهٔ ستاپ: " + pat_line(b))
    else:
        parts.append("خانوادهٔ ستاپ: داده‌ی کافی ندارد")
    refined = m[m["cond"].apply(len) > 0]
    best = refined[refined["tier"] != "none"].drop_duplicates(["n_tr", "mean_tr", "n_va"]).head(2)
    for _, r in best.iterrows():
        parts.append("الگوی منطبق (فرضیهٔ فرعی، احتمال نویز): " + pat_line(r))
    if best.empty:
        parts.append("الگوی دقیق‌تر منطبق با اعتبار متوسط/ضعیف: ندارد")
    tier = "medium" if (m["tier"] == "medium").any() else base_tier      # weak sub-patterns never lift the verdict: they are the likeliest noise
    return parts, tier, base_tier


def level_block(df, row, digits=2):
    e = int(row["entry_i"])
    entry = float(df["open"].iloc[e])
    s, d, rr = row["side"], row["dist"], row["rr"]
    stop, tgt = entry - s * d, entry + s * rr * d
    return {"entry": round(entry, digits), "stop": round(stop, digits), "target": round(tgt, digits), "dist": round(d, digits), "atr_x": round(d / row["atr"], 2), "rr": rr}


def block_text(row, ev_parts, lv, sym="XAUUSD"):
    f = [f"{tp.FEAT_FA[k]}: {VAL(row[k])}" for k in ("reg", "vol", "trend", "ext", "rsi")]
    return ("[ستاپ مکانیکی و شواهد تاریخی]\n"
            f"ستاپ: {tp.FA[row['strategy']]} | جهت: {SIDE_FA[row['side']]} | ورود: باز شدن کندل بعدی نزدیک {lv['entry']} | حد ضرر: {lv['stop']} (فاصله {lv['dist']} = {lv['atr_x']} ATR) | هدف: {lv['target']} (ریوارد {lv['rr']})\n"
            "ویژگی‌های لحظه: " + "؛ ".join(f) + "\n" + "\n".join(ev_parts) +
            "\nروش: معاملات مکانیکی با هزینه (اسپرد+لغزش)، الگوها فقط روی بازهٔ آموزش استخراج و با FDR اصلاح و روی بازهٔ بعدی تأیید شده‌اند؛ این شواهد پیش‌بینی نیستند.")


def struct_line(row, ctx):
    m = re.search(r"روند ساختاری: (صعودی|نزولی)", ctx)
    if not m:
        return "روند ساختاری در داده‌ها مشخص نیست."
    same = (m.group(1) == "صعودی") == (row["side"] == 1)
    return f"جهت ستاپ {'هم‌جهت با' if same else 'خلاف'} روند ساختاری {m.group(1)} است" + ("." if same else "؛ این ورود خلاف ساختار است و ریسک بیشتری دارد.")


def eval_answer(rng, row, lv, tier, base_tier, ev_parts, ctx=""):
    base = ev_parts[0].split("خانوادهٔ ستاپ: ", 1)[1]
    verdict = {"medium": "شواهد تاریخی متوسط‌اند؛ می‌توان با ریسک کوچک و کنترل‌شده بررسی کرد، نه بیشتر.",
               "weak": "شواهد ضعیف‌اند و ممکن است نویز باشند؛ بیشتر برای ثبت آزمایشی یا ریسک خیلی کوچک مناسب است و اولویت با صبر کردن است.",
               "none": "شواهد تاریخی کافی برای این ستاپ نداریم؛ پیشنهاد روشن این است که نگیرید یا فقط روی حساب دمو/ثبت آزمایشی دنبال کنید."}[tier]
    lose = 100 - round(float(base.split("برد ")[1].split("%")[0])) if "برد " in base else None
    lines = [f"ستاپ {tp.FA[row['strategy']]} ({SIDE_FA[row['side']]}) به‌صورت مکانیکی فعال شده است.", struct_line(row, ctx),
             "شواهد: " + " / ".join(p.split(": ", 1)[1] for p in ev_parts[:3]),
             f"جمع‌بندی: {verdict}",
             f"سطوح قاعده: ورود نزدیک {lv['entry']}، حد ضرر {lv['stop']}، هدف {lv['target']}. اگر قیمت حد ضرر را بشکند سناریو باطل است و بدون جابه‌جایی حد ضرر خارج می‌شوید."]
    if lose is not None:
        lines.append(f"یادآوری: حتی در تاریخچهٔ همین خانواده حدود {lose}% معاملات ضرر بوده‌اند؛ هر معاملهٔ تکی می‌تواند ضرر باشد و این پیش‌بینی نتیجهٔ همین معامله نیست.")
    lines.append(DISCLAIMER)
    return "\n\n".join(lines), ([str(lose)] if lose is not None else [])


def review_answer(row, tier, base_tier, ev_parts, won):
    base = ev_parts[0].split("خانوادهٔ ستاپ: ", 1)[1]
    good = tier in ("medium", "weak") and row["split"] != "x"
    if tier == "none":
        dq = "از نظر شواهد، ورود تصمیم ضعیفی بود؛ برای این ستاپ شواهد کافی نداشتیم."
    elif tier == "weak":
        dq = "ورود با شواهد ضعیف قابل‌دفاع بود فقط اگر ریسک بسیار کوچک بوده باشد؛ شواهد قوی نبودند."
    else:
        dq = "ورود با شواهد متوسط و ریسک کوچک قابل‌دفاع بود."
    if won:
        out = "نتیجه برنده بود، اما یک برد کیفیت تصمیم را ثابت نمی‌کند" + ("؛ برای تصمیمی که شواهد کافی نداشت این بیشتر شانس است." if tier == "none" else ".")
    else:
        out = "نتیجه ضرر بود، اما یک ضرر به‌تنهایی نشان‌دهندهٔ تصمیم بد نیست" + ("، هرچند اینجا شواهد هم ضعیف بود." if tier == "none" else "؛ در این خانواده ضرر بخش عادی نتایج است.")
    lesson = "درس: تصمیم را با شواهد و قواعد ریسک بسنجید، نه با نتیجهٔ یک معامله. الگو را فقط با ده‌ها معاملهٔ جدید و جدا از دادهٔ استخراج می‌توان آزمود، نه با این یک مورد."
    txt = f"{dq}\n\nشواهد آن زمان: {base}\n\n{out}\n\n{lesson}\n\n{DISCLAIMER}"
    return txt


WHY = ("چون برچسب «معامله‌ی درست» پس از دیدن آینده ساخته می‌شود و مدل در لحظهٔ واقعی آینده را نمی‌بیند. اگر فقط برنده‌ها را نشان بدهیم، مدل شکست‌ها و فراوانی آن‌ها را نمی‌بیند "
       "و اعتماد به‌نفس بی‌پایه می‌گیرد (سوگیری بقا و نگاه‌به‌گذشته). راه درست این است: همهٔ معاملات مکانیکی، چه برنده و چه بازنده، با هزینه واقعی ثبت شوند؛ الگوها فقط روی بازهٔ آموزش استخراج، "
       "با اصلاح چندآزمونی (FDR) سنجیده و روی دادهٔ بعدی تأیید شوند؛ و مدل یاد بگیرد شواهد را بخواند، اعتبار را (متوسط/ضعیف/هیچ) بگوید و به‌جای پیش‌بینی، سناریو و ریسک بدهد. "
       "هم‌چنین هر الگو را باید روی دادهٔ مهرشده سنجید. اگر هیچ الگویی از این غربال عبور نکند، پاسخ درست «لبه‌ای ثابت نشده» است. " + DISCLAIMER)


def table_text(P, k=8):
    top = P[P["tier"] != "none"].sort_values(["tier", "p_tr"], ascending=[True, True])
    top = top[top["side"] != "any"].head(k)
    L = [f"تعداد الگوهای آزمون‌شده: {int(P['n_tested'].iloc[0])} | با اعتبار متوسط: {int((P['tier'] == 'medium').sum())} | با اعتبار ضعیف: {int((P['tier'] == 'weak').sum())}"]
    L += [pat_line(r) for _, r in top.iterrows()]
    return "[جدول الگوهای تاریخی (آموزش ۲۰۲۱-۰۹ تا ۲۰۲۴-۰۶، تأیید تا ۲۰۲۵-۰۶)]\n" + "\n".join(L), top


def table_answer(P, top):
    nm, nw = int((P["tier"] == "medium").sum()), int((P["tier"] == "weak").sum())
    fams = sorted({tp.FA[s] for s in top["strategy"]})
    parts = []
    if nm == 0:
        parts.append("هیچ الگویی پس از اصلاح چندآزمونی به اعتبار متوسط نرسیده است.")
    else:
        parts.append(f"{nm} الگو به اعتبار متوسط رسیده‌اند.")
    parts.append(f"{nw} الگو فقط اعتبار ضعیف دارند، یعنی روی آموزش معنادار اسمی بوده‌اند و در بازهٔ تأیید هم مثبت ماندند؛ ممکن است نویز باشند.")
    if len(top):
        parts.append("بیشترین تکرار در خانواده‌های: " + "، ".join(fams) + ". بسیاری از ردیف‌ها همپوشان‌اند (یک اثر در چند شرط) و مستقل حساب نمی‌شوند.")
    parts.append("نتیجه: چیزی فراتر از «فرضیهٔ ضعیف برای آزمون رو‌به‌جلو» ثابت نشده است؛ پیشنهاد می‌شود آن‌ها را روی داده یا حساب دمو جدید پیگیری کنید و با ریسک کوچک وارد نشوید.")
    parts.append(DISCLAIMER)
    return "\n\n".join(parts)


def build(out_dir, candles_dir, seed=2028, n=None, symbol="XAUUSD"):
    n = n or {"train": 1000, "val": 180, "test": 180}
    rng = random.Random(seed)
    df = el.load_csv(candles_dir, symbol)
    T = tp.trades_table(df)
    P = tp.mine(T)
    candles = v1.load_candles(candles_dir, symbol)
    R = v2.Rows()
    made = {}
    for split in ("train", "val", "test"):
        S = T[T["split"] == split].sample(frac=1, random_state=rng.randint(0, 10 ** 6))
        last, cnt = [], 0
        for _, row in S.iterrows():
            if cnt >= n[split]:
                break
            t = pd.Timestamp(int(row["entry_epoch"]), unit="s")
            if any(abs((t - u).total_seconds()) < 5 * 3600 for u in last):
                continue
            data, info = v1.make_context(candles, t, smc=True, symbol=symbol, digits=2, spread=0.2)
            if data is None:
                continue
            last.append(t)
            cnt += 1
            ev_parts, tier, base_tier = evidence_block(P, row)
            lv = level_block(df, row)
            full = data + "\n\n" + block_text(row, ev_parts, lv, symbol)
            meta = {"t": info["t"], "te": info["te"], "last_open": info["last_open"], "symbol": symbol, "strategy": row["strategy"], "side": int(row["side"]), "tier": tier,
                    "derived": []}
            sysm = v1.sysm(rng, "data", data=full)
            ans, der = eval_answer(rng, row, lv, tier, base_tier, ev_parts, data)
            meta_e = dict(meta, derived=der)
            R.add_msgs("trades_v3_eval", "setup." + tier, "setup_eval", split, [{"role": "system", "content": sysm},
                     {"role": "user", "content": rng.choice(ASK_EVAL).format(s=tp.FA[row["strategy"]], sym=symbol)}, {"role": "assistant", "content": ans}],
                     ev={"must_include_any_of_each": [["آموزشی"], ["حد ضرر"]], "must_not_match": [r"تضمین می‌کنم|حتماً برنده|قطعاً"]}, meta=meta_e)
            if rng.random() < 0.8:
                won = row["r"] > 0
                outcome = f"{'سود' if won else 'ضرر'} {fmt_r(row['r'])} پس از {int(row['exit_i'] - row['entry_i'] + 1)} کندل"
                q = rng.choice(ASK_REVIEW).format(s=tp.FA[row["strategy"]], o=outcome)
                a = review_answer(row, tier, base_tier, ev_parts, won)
                R.add_msgs("trades_v3_review", f"review.{'win' if won else 'loss'}.{tier}", "post_trade_review", split,
                           [{"role": "system", "content": sysm}, {"role": "user", "content": q}, {"role": "assistant", "content": a}],
                           ev={"must_include_any_of_each": [["آموزشی"], ["شواهد"]], "must_not_match": [r"تضمین می‌کنم|قطعاً"]}, meta=dict(meta, won=bool(won), r=round(float(row["r"]), 2), derived=[]))
        made[split] = cnt
    for i in range(36):
        tt, top = table_text(P, k=rng.choice([4, 6, 8]))
        sysm = v1.sysm(rng, "data", data=tt)
        sp = "train" if i < 30 else "val" if i < 33 else "test"
        R.add_msgs("trades_v3_table", "table.summary", "pattern_table", sp, [{"role": "system", "content": sysm}, {"role": "user", "content": rng.choice(ASK_TABLE)},
                   {"role": "assistant", "content": table_answer(P, top)}], ev={"must_include_any_of_each": [["آموزشی"]], "must_not_match": [r"تضمین می‌کنم"]}, meta={"derived": [], "symbol": symbol})
    for i, q in enumerate(ASK_WHY * 4):
        sp = "train" if i < 10 else "val" if i < 11 else "test"
        R.add_msgs("trades_v3_why", "why.not_winners_only", "method", sp, [{"role": "system", "content": v1.sysm(rng, "none")}, {"role": "user", "content": q}, {"role": "assistant", "content": WHY}],
                   ev={"must_include_any_of_each": [["آموزشی"]], "must_not_match": []}, meta={"derived": []})
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    for task, rows in R.rows.items():
        seen, uniq = set(), []
        for r in rows:
            if r["id"] not in seen:
                seen.add(r["id"]); uniq.append(r)
        uniq.sort(key=lambda r: r["id"])
        p = os.path.join(out_dir, task + ".jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in uniq:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        c: Dict[str, int] = {}
        for r in uniq:
            c[r["split"]] = c.get(r["split"], 0) + 1
        files[task + ".jsonl"] = {"rows": len(uniq), "split": c}
    P2 = P.copy(); P2["cond"] = P2["cond"].astype(str)
    P2.to_csv(os.path.join(out_dir, "pattern_table.csv"), index=False)
    T.drop(columns=[]).to_csv(os.path.join(out_dir, "trades_all.csv"), index=False)
    rep = {"dataset": "aurora_trades_v3", "symbol": symbol, "trades": int(len(T)), "by_split": T["split"].value_counts().to_dict(), "patterns_tested": int(len(P)),
           "tiers": P["tier"].value_counts().to_dict(), "tier_vs_test": tp_tier_check(P), "contexts": made, "files": files, "trained": False}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    return rep


def tp_tier_check(P):
    res = {}
    Q = P[(P["side"] != "any") & (P["n_te"] >= 20)]
    for t, g in Q.groupby("tier"):
        res[t] = {"patterns": int(len(g)), "share_positive_in_test": round(float((g["mean_te"] > 0).mean()), 3), "median_test_mean_R": round(float(g["mean_te"].median()), 3)}
    return res


def export_sft(src_dir, out_dir):
    rows = []
    for name in sorted(os.listdir(src_dir)):
        if name.endswith(".jsonl") and name.startswith("trades_v3"):
            with open(os.path.join(src_dir, name), encoding="utf-8") as f:
                rows += [json.loads(l) for l in f if l.strip()]
    os.makedirs(out_dir, exist_ok=True)
    c = {}
    for sp in ("train", "val", "test"):
        part = sorted((r for r in rows if r["split"] == sp), key=lambda r: r["id"])
        with open(os.path.join(out_dir, f"trades_v3_sft_{sp}.jsonl"), "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps({"id": r["id"], "messages": r["messages"]}, ensure_ascii=False) + "\n")
        c[sp] = len(part)
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="training_data/aurora_trades_v3")
    ap.add_argument("--candles", default="/home/claude/work/XAUUSD_Dataset/out/01_normalized")
    a = ap.parse_args()
    m = build(a.out, a.candles)
    print(json.dumps({k: m[k] for k in ("trades", "by_split", "patterns_tested", "tiers", "tier_vs_test", "contexts")}, ensure_ascii=False, indent=1))
    print({k: v["rows"] for k, v in m["files"].items()}, "sft:", export_sft(a.out, os.path.join(a.out, "sft")))
