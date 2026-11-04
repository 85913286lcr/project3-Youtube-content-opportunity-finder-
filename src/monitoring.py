"""Monitoring simulation over the 4 holdout weeks.

The topic model is frozen on the train period. Each week new comments arrive and
we check whether the model still describes them:

    need_share        share of comments the classifier flags as a need
    assigned_rate     share of need comments that fit an existing topic
    novelty_rate      share of need comments Isolation Forest (L05) calls unusual
                      compared with the train-period need comments
    topic_shift       how different this week's topic mix is from the train mix
                      (total variation distance, 0 = same, 1 = no overlap)

Alert rules (set from the train weeks, see notebook 05):
    assigned_rate below 75%  or  novelty_rate above 2x the train level
    -> look at the unusual comments, re-cluster and re-name topics

The unusual comments of the latest week are also clustered into a few groups so
a person can see what new subjects are appearing.

Usage:
    python -m src.monitoring
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from src import config
from src.clustering import top_keywords
from src.embed import load_comment_embeddings
from src.preprocess import TRAIN_END

OUT = config.DATA_PROCESSED
RS = config.RANDOM_STATE


def run():
    c = pd.read_csv(OUT / "comments_topics.csv", parse_dates=["published_at"])
    emb = load_comment_embeddings()
    c["week"] = c["published_at"].dt.tz_localize(None).dt.to_period("W-SUN").dt.start_time

    need = c["is_need"] == 1
    tr = (need & (c["period"] == "train")).values

    # Isolation Forest on the train need comments, in PCA space (fitted on train only)
    pca = PCA(50, random_state=RS).fit(emb[tr])
    iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=RS).fit(
        pca.transform(emb[tr]))
    c["novelty_score"] = np.nan
    c.loc[need, "novelty_score"] = iso.score_samples(pca.transform(emb[need.values]))
    cutoff = np.quantile(c.loc[tr, "novelty_score"], 0.05)
    c["is_novel"] = need & (c["novelty_score"] < cutoff)

    train_mix = c[tr & (c["topic"] >= 0).values]["topic"].value_counts(normalize=True)

    rows = []
    weeks = sorted(c["week"].unique())
    for w in weeks[-9:]:                      # last 4 train weeks + 4 holdout weeks
        wk = c[c["week"] == w]
        if len(wk) < 500:                     # the final week only has one day of data
            continue
        wn = wk[wk["is_need"] == 1]
        mix = wn[wn["topic"] >= 0]["topic"].value_counts(normalize=True)
        tv = 0.5 * mix.reindex(train_mix.index.union(mix.index), fill_value=0).sub(
            train_mix.reindex(train_mix.index.union(mix.index), fill_value=0)).abs().sum()
        rows.append({"week": w.date(), "period": "holdout" if w >= TRAIN_END.tz_localize(None) else "train",
                     "comments": len(wk), "need_comments": len(wn),
                     "need_share": len(wn) / max(len(wk), 1),
                     "assigned_rate": (wn["topic"] >= 0).mean(),
                     "novelty_rate": wn["is_novel"].mean(),
                     "topic_shift": tv})
    mon = pd.DataFrame(rows)
    base_novelty = mon.loc[mon["period"] == "train", "novelty_rate"].mean()
    mon["alert"] = (mon["assigned_rate"] < 0.75) | (mon["novelty_rate"] > 2 * base_novelty)
    mon.to_csv(OUT / "monitoring_weekly.csv", index=False)

    # what is new in the holdout weeks? group the unusual comments
    recent = c[(c["period"] == "holdout") & c["is_novel"]]
    groups = pd.DataFrame()
    if len(recent) >= 20:
        k = 5
        lab = KMeans(k, n_init=10, random_state=RS).fit_predict(emb[recent.index])
        kw = top_keywords(recent["text_clean"].to_numpy(dtype=object), lab, top_n=6)
        groups = (recent.assign(group=lab).groupby("group")
                  .agg(n=("comment_id", "size"),
                       example=("text_clean", lambda s: " || ".join(s.head(3).str[:140])))
                  .reset_index())
        groups["keywords"] = groups["group"].map(kw)
        groups.sort_values("n", ascending=False).to_csv(OUT / "monitoring_new_themes.csv", index=False)
    c[need][["comment_id", "novelty_score", "is_novel"]].to_csv(OUT / "comment_novelty.csv", index=False)

    pd.set_option("display.width", 200)
    print(mon.round(3).to_string(index=False))
    print("train novelty baseline:", round(base_novelty, 3))
    if not groups.empty:
        print(groups[["group", "n", "keywords"]].to_string(index=False))


if __name__ == "__main__":
    run()
