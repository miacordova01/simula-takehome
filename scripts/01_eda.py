"""
EDA pass over the Simula dataset.

Goal: understand structure, cardinality, join integrity, signal in the
character/conversation columns, and temporal behaviour -- before choosing a model.

Writes reports/eda.md and a few small CSVs into reports/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simula.config import CHARACTERS_CSV, IMPRESSIONS_CSV, REPORTS  # noqa: E402
from simula.data import load_characters, load_impressions  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 100)

OUT: list[str] = []


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def section(title: str) -> None:
    say()
    say(f"## {title}")
    say()


def table(df: pd.DataFrame, floatfmt: str = "%.4f") -> None:
    say("```")
    say(df.to_string(float_format=lambda x: floatfmt % x))
    say("```")


def main() -> None:
    say("# Simula dataset EDA")
    say()
    # Filenames only -- absolute paths would bake this machine's home directory
    # into a committed report.
    say(f"- impressions: `{IMPRESSIONS_CSV.name}`")
    say(f"- characters:  `{CHARACTERS_CSV.name}`")

    imp = load_impressions()
    chars = load_characters()

    # ------------------------------------------------------------------
    section("Shape and target")
    say(f"- impressions rows: {len(imp):,}, cols: {imp.shape[1]}")
    say(f"- characters rows:  {len(chars):,}, cols: {chars.shape[1]}")
    ctr = imp["click"].mean()
    say(f"- base CTR: **{ctr:.4%}** ({int(imp['click'].sum()):,} clicks)")
    say(f"- unique impression ids: {imp['id'].nunique():,} (dupes: {len(imp) - imp['id'].nunique():,})")
    say(f"- null counts total: {int(imp.isna().sum().sum())}")

    # ------------------------------------------------------------------
    section("Cardinality of every column")
    rows = []
    for c in imp.columns:
        if c in ("id",):
            continue
        nu = imp[c].nunique()
        top = imp[c].value_counts(normalize=True)
        rows.append(
            {
                "column": c,
                "n_unique": nu,
                "top_value": str(top.index[0]),
                "top_share": top.iloc[0],
                "top10_share": top.iloc[:10].sum(),
                "singletons": int((imp[c].value_counts() == 1).sum()),
            }
        )
    card = pd.DataFrame(rows).sort_values("n_unique", ascending=False)
    table(card)

    # ------------------------------------------------------------------
    section("Temporal structure")
    ts = imp["hour"].astype(str)
    imp["_day"] = ts.str[:6]
    imp["_hod"] = ts.str[6:8].astype(int)
    imp["_dt"] = pd.to_datetime(ts, format="%y%m%d%H")
    imp["_dow"] = imp["_dt"].dt.dayofweek

    day = imp.groupby("_day").agg(n=("click", "size"), ctr=("click", "mean"))
    day["share"] = day["n"] / len(imp)
    say("Per-day volume and CTR:")
    table(day)

    hod = imp.groupby("_hod").agg(n=("click", "size"), ctr=("click", "mean"))
    say("Per-hour-of-day CTR:")
    table(hod)

    dow = imp.groupby("_dow").agg(n=("click", "size"), ctr=("click", "mean"))
    say("Per-day-of-week CTR (0=Mon):")
    table(dow)

    say(f"- full time range: {imp['_dt'].min()} .. {imp['_dt'].max()}")
    last_day_hours = sorted(imp.loc[imp["_day"] == day.index[-1], "_hod"].unique())
    say(f"- last day ({day.index[-1]}) covers hours: {last_day_hours}")

    # ------------------------------------------------------------------
    section("Character join")
    ch_ids = set(chars["character_id"])
    imp_ids = set(imp["character_id"].unique())
    matched = imp["character_id"].isin(ch_ids)
    say(f"- distinct characters in impressions: {len(imp_ids):,}")
    say(f"- distinct characters in characters.csv: {len(ch_ids):,}")
    say(f"- impressions rows joining successfully: {matched.mean():.4%}")
    say(f"- characters in impressions but NOT in characters.csv: {len(imp_ids - ch_ids):,}")
    say(f"- characters in characters.csv never served: {len(ch_ids - imp_ids):,}")

    vc = imp["character_id"].value_counts()
    say(f"- impressions per character: mean={vc.mean():.1f} median={vc.median():.0f} "
        f"p95={vc.quantile(0.95):.0f} max={vc.max():,}")
    say(f"- top-10 characters cover {vc.iloc[:10].sum() / len(imp):.2%} of impressions")
    say(f"- top-100 characters cover {vc.iloc[:100].sum() / len(imp):.2%} of impressions")
    say(f"- characters with <10 impressions: {(vc < 10).sum():,} "
        f"({(vc < 10).sum() / len(vc):.1%} of served characters)")

    # ------------------------------------------------------------------
    section("Character attributes vs CTR")
    j = imp.merge(chars, on="character_id", how="left", validate="m:1")

    for col in ["safety_tier", "creator_type"]:
        g = j.groupby(col, dropna=False).agg(n=("click", "size"), ctr=("click", "mean"))
        g["lift"] = g["ctr"] / ctr
        say(f"CTR by `{col}`:")
        table(g)

    j["_ni_bucket"] = pd.qcut(j["num_interactions"], 10, duplicates="drop")
    g = j.groupby("_ni_bucket", observed=True).agg(n=("click", "size"), ctr=("click", "mean"))
    g["lift"] = g["ctr"] / ctr
    say("CTR by `num_interactions` decile:")
    table(g)

    j["_created"] = pd.to_datetime(j["created_at"])
    j["_char_age_days"] = (j["_dt"] - j["_created"]).dt.days
    say(f"- character age (days at impression time): min={j['_char_age_days'].min():.0f} "
        f"median={j['_char_age_days'].median():.0f} max={j['_char_age_days'].max():.0f}")
    j["_age_bucket"] = pd.qcut(j["_char_age_days"], 10, duplicates="drop")
    g = j.groupby("_age_bucket", observed=True).agg(n=("click", "size"), ctr=("click", "mean"))
    g["lift"] = g["ctr"] / ctr
    say("CTR by character-age decile:")
    table(g)

    # per-character CTR dispersion -- is character_id itself predictive?
    pc = imp.groupby("character_id")["click"].agg(["size", "mean"])
    big = pc[pc["size"] >= 200]
    say(f"- characters with >=200 impressions: {len(big):,}")
    say(f"- their CTR spread: p05={big['mean'].quantile(0.05):.4f} "
        f"p50={big['mean'].median():.4f} p95={big['mean'].quantile(0.95):.4f}")
    say("  (wide spread => character_id carries real signal; narrow => mostly noise)")

    # ------------------------------------------------------------------
    section("Conversation-state features")
    say(f"- conversation_turn: min={imp['conversation_turn'].min()} "
        f"median={imp['conversation_turn'].median():.0f} p95={imp['conversation_turn'].quantile(0.95):.0f} "
        f"max={imp['conversation_turn'].max()}")
    say(f"- session_msg_count: min={imp['session_msg_count'].min()} "
        f"median={imp['session_msg_count'].median():.0f} p95={imp['session_msg_count'].quantile(0.95):.0f} "
        f"max={imp['session_msg_count'].max()}")
    viol = (imp["conversation_turn"] > imp["session_msg_count"]).mean()
    say(f"- rows where turn > session_msg_count: {viol:.4%}")

    turn_cap = imp["conversation_turn"].clip(upper=15)
    g = imp.groupby(turn_cap).agg(n=("click", "size"), ctr=("click", "mean"))
    g["lift"] = g["ctr"] / ctr
    say("CTR by conversation_turn (capped at 15):")
    table(g)

    smc_cap = imp["session_msg_count"].clip(upper=25)
    g = imp.groupby(smc_cap).agg(n=("click", "size"), ctr=("click", "mean"))
    g["lift"] = g["ctr"] / ctr
    say("CTR by session_msg_count (capped at 25):")
    table(g)

    prog = (imp["conversation_turn"] / imp["session_msg_count"].clip(lower=1)).clip(0, 1)
    g = imp.groupby(pd.cut(prog, 10), observed=True).agg(n=("click", "size"), ctr=("click", "mean"))
    g["lift"] = g["ctr"] / ctr
    say("CTR by session progress (turn / session_msg_count):")
    table(g)

    # ------------------------------------------------------------------
    section("Device / user identity")
    for col in ["device_id", "device_ip", "device_model"]:
        vc2 = imp[col].value_counts()
        say(f"`{col}`: {len(vc2):,} unique, top='{vc2.index[0]}' share={vc2.iloc[0]/len(imp):.2%}, "
            f"singleton share={(vc2 == 1).sum()/len(imp):.2%}")
    dom = imp["device_id"].value_counts().index[0]
    say(f"- dominant device_id '{dom}' is the classic 'unknown/null' sentinel; "
        f"CTR on it = {imp.loc[imp['device_id']==dom,'click'].mean():.4f} "
        f"vs {imp.loc[imp['device_id']!=dom,'click'].mean():.4f} elsewhere")

    # repeat exposure -- how often do we see the same device more than once?
    dv = imp["device_ip"].value_counts()
    say(f"- device_ip appearing >1 time: {(dv > 1).sum():,} ips covering "
        f"{dv[dv > 1].sum()/len(imp):.1%} of impressions (=> fatigue caps are meaningful)")

    # ------------------------------------------------------------------
    section("Anonymized C-features vs CTR")
    for c in ["C1", "banner_pos", "C15", "C16", "C18", "C19", "C21"]:
        g = imp.groupby(c).agg(n=("click", "size"), ctr=("click", "mean"))
        g = g[g["n"] > len(imp) * 0.001].sort_values("n", ascending=False).head(12)
        g["lift"] = g["ctr"] / ctr
        say(f"CTR by `{c}` (buckets >0.1% of traffic):")
        table(g)

    # ------------------------------------------------------------------
    section("Drift preview: are the same characters served across days?")
    top_by_day = {}
    for d, sub in imp.groupby("_day"):
        top_by_day[d] = set(sub["character_id"].value_counts().index[:200])
    days = sorted(top_by_day)
    jac = pd.DataFrame(index=days, columns=days, dtype=float)
    for a in days:
        for b in days:
            inter = len(top_by_day[a] & top_by_day[b])
            union = len(top_by_day[a] | top_by_day[b])
            jac.loc[a, b] = inter / union
    say("Jaccard overlap of each day's top-200 characters:")
    table(jac, "%.3f")

    say()
    say("First-vs-last-day overlap of top-200 characters: "
        f"**{jac.loc[days[0], days[-1]]:.3f}** -- lower means faster character churn.")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "eda.md").write_text("\n".join(OUT))
    card.to_csv(REPORTS / "cardinality.csv", index=False)
    print(f"\nWrote {REPORTS/'eda.md'}")


if __name__ == "__main__":
    main()
