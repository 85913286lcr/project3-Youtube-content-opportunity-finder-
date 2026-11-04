"""Backtest of the demand side: do the topics picked on a date keep being asked
about in the 4 weeks after?

At each cutoff date, using only data up to that date, each strategy picks 3
topics per niche:

    Opportunity score   0.45 D + 0.35 G + 0.20 F, Low-confidence topics excluded
    Most asked          top 3 by distinct askers only
    Random              expected value of a random pick among actionable topics

Outcome: distinct askers per topic in the 28 days after the cutoff.
    hit rate   share of picks that end up in the top half of their niche
    spearman   rank correlation between askers before and after (all topics)

Cutoffs (rolling origin, L06):
    3 Aug to 14 Sep  development checks, one per week (7 dates). The earliest date
                     is 3 Aug because each pick needs 30 days of comments before it
                     and the data starts on 28 Jun 2026.
    28 Sep           final check, reported once. This date was also used in the
                     first version of the project, so it is not fully unseen.

Views are not measured here; the A/B test in docs/11_deployment_plan.md does that.

Usage:
    python -m src.backtest
"""
import numpy as np
import pandas as pd

from src import config
from src.scoring import Coverage, load_data, score_topics

OUT = config.DATA_PROCESSED
DEV_CUTOFFS = ["2026-08-03", "2026-08-10", "2026-08-17", "2026-08-24", "2026-08-31", "2026-09-07", "2026-09-14"]
CUTOFFS = [(c, "development") for c in DEV_CUTOFFS] + [("2026-09-28", "final")]


def future_askers(comments, topics, niche, start, days=28):
    end = start + pd.Timedelta(days=days)
    act = topics[(topics["niche"] == niche) & (topics["actionable"] == 1)]["topic"]
    c = comments[(comments["niche"] == niche) & (comments["topic"].isin(act))
                 & (comments["published_at"] > start) & (comments["published_at"] <= end)]
    return c.groupby("topic")["author_hash"].nunique().reindex(act).fillna(0)


def run():
    comments, videos, topics, videos_all = load_data()
    cov = Coverage(comments, videos_all)
    rows, picks = [], []
    for cut, kind in CUTOFFS:
        T = pd.Timestamp(cut, tz="UTC")
        for niche in config.NICHES:
            s = score_topics(comments, videos, topics, niche, T, cov)
            fut = future_askers(comments, topics, niche, T)
            med = fut.median()
            model_pick = list(s[s["eligible"]]["topic"].head(3))
            asked_pick = list(s.sort_values("n_users", ascending=False)["topic"].head(3))
            hit = lambda tp: float(np.mean([fut.get(t, 0) > med for t in tp])) if tp else np.nan
            rho = s.set_index("topic")["n_users"].corr(fut.reindex(s["topic"]).set_axis(s["topic"]),
                                                       method="spearman")
            rows += [
                {"cutoff": cut, "kind": kind, "niche": niche, "strategy": "Opportunity score",
                 "hit_rate": hit(model_pick), "spearman": rho},
                {"cutoff": cut, "kind": kind, "niche": niche, "strategy": "Most asked",
                 "hit_rate": hit(asked_pick), "spearman": rho},
                {"cutoff": cut, "kind": kind, "niche": niche, "strategy": "Random",
                 "hit_rate": float((fut > med).mean()), "spearman": rho},
            ]
            names = topics.set_index("topic")["topic_name"]
            picks.append({"cutoff": cut, "niche": niche, "picks": [names[t] for t in model_pick]})
    res = pd.DataFrame(rows)
    summary = (res.groupby(["kind", "cutoff", "strategy"], sort=False)
               .agg(hit_rate=("hit_rate", "mean"), spearman=("spearman", "mean")).reset_index())
    res.to_csv(OUT / "backtest_by_niche.csv", index=False)
    summary.to_csv(OUT / "backtest_summary.csv", index=False)
    pd.DataFrame(picks).to_csv(OUT / "backtest_picks.csv", index=False)
    old = OUT / "backtest_persistence.csv"
    if old.exists():
        old.unlink()
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    run()
