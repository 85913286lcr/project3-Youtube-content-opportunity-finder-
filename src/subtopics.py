"""Specific angles: a second, finer clustering layer inside each niche.

The broad topics (15-20 per niche) stay the unit for ranking, the backtest and
monitoring. They are wide on purpose: wide enough to collect evidence from many
people. A creator then needs a narrower answer: inside "Using AI for job
applications and interviews", is the video about AI interviews or about auto-apply?

Method (L05, same MiniLM embeddings as src/clustering.py), per niche, train period only:
    1. Consensus K-Means: 20 K-Means runs with different seeds, K = questions / 15.
       Two questions are "together" in the share of runs that put them in the same
       cluster. A single K-Means run at this size is unstable (ARI 0.32-0.42 between
       seeds): with ~15 questions per cluster the boundaries move with the seed.
    2. Hierarchical clustering (average linkage) on that together-share, cut at 0.5:
       questions that land together in at least half of the runs form an angle's
       core. Cores under 5 questions are dropped.
    3. Merge look-alike angles: centres more similar than 0.80 (cosine) become one,
       so a big question ("what is an AI agent?") is not cut into near-copies.
    4. Every need comment (train + holdout) joins its nearest angle if the cosine is
       at least 0.45, otherwise it keeps only its broad topic. 0.45 keeps about 80% of
       questions, close to the broad-topic fit rate; vaguer ones ("make more videos")
       stay without an angle.
    5. An angle hangs under the broad topic holding most of its train questions, and
       is also listed under any filmable topic holding at least 25% of them. Angles
       under a topic flagged not filmable are hidden.
    6. TF-IDF keywords + the 2 questions closest to the centre describe each angle.

Stability: the whole procedure is repeated with two other sets of 20 seeds and
compared with the adjusted Rand index; per angle, the best Jaccard overlap tells
whether the same questions come back together (>= 0.5 = stable).

K is set by size rather than by silhouette: silhouette picks K = 2 inside a topic,
which is not specific enough to film. Angles are a reading aid for the creator,
not a new ranking, so none of the evaluated numbers change.

Usage:
    python -m src.subtopics
"""
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score

from src import config
from src.clustering import NICHE_OFFSET, top_keywords
from src.embed import load_comment_embeddings

OUT = config.DATA_PROCESSED
NAMES = config.DATA_LABELS / "subtopic_names.csv"
TARGET_SIZE = 15                  # K = questions / 15 in each K-Means run
N_RUNS = 20                       # K-Means runs per consensus
TOGETHER = 0.5                    # same angle if together in at least half of the runs
MIN_CORE = 5                      # smaller cores are dropped
MERGE_SIM = 0.80                  # look-alike angles are merged
GUARD_SIM = 0.45                  # a question joins its nearest angle only if this close
ALSO_SHARE = 0.25
RECENT_DAYS = 30
SEED_SETS = [range(0, 20), range(100, 120), range(200, 220)]   # the first one is kept
STABLE = 0.5                      # an angle whose questions mostly come back together


def unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def fit_angles(X, seeds):
    """Consensus K-Means. Returns angle centres, train labels (-1 = no angle), K and the guard."""
    k = max(10, round(len(X) / TARGET_SIZE))
    A = np.zeros((len(X), len(X)))
    for s in seeds:
        lab = KMeans(n_clusters=k, n_init=3, random_state=s).fit_predict(X)
        A += lab[:, None] == lab[None, :]
    A /= len(seeds)
    core = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                   distance_threshold=1 - TOGETHER).fit_predict(1 - A)
    keep = [c for c in np.unique(core) if (core == c).sum() >= MIN_CORE]
    C = unit(np.vstack([X[core == c].mean(0) for c in keep]))
    merged = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=1 - MERGE_SIM).fit_predict(C)
    groups = [np.isin(core, [keep[i] for i in np.where(merged == g)[0]]) for g in np.unique(merged)]
    centres = unit(np.vstack([X[m].mean(0) for m in groups]))
    guard = GUARD_SIM
    sims = X @ centres.T
    labels = np.where(sims.max(1) >= guard, sims.argmax(1), -1)
    return centres, labels, k, guard


def stability(X, base):
    """ARI between seed sets, and per angle of `base` the mean best Jaccard overlap
    with any angle of each other seed set (1 = the same questions come back)."""
    labs = [base] + [fit_angles(X, s)[1] for s in SEED_SETS[1:]]
    aris = []
    for a, b in combinations(labs, 2):
        both = (a >= 0) & (b >= 0)
        aris.append(adjusted_rand_score(a[both], b[both]))
    per = {}
    for c in np.unique(base[base >= 0]):
        m = base == c
        per[int(c)] = float(np.mean([max(((m & (l == d)).sum() / (m | (l == d)).sum())
                                         for d in np.unique(l[m]) if d >= 0) if (l[m] >= 0).any() else 0.0
                                     for l in labs[1:]]))
    return float(np.mean(aris)), per


def run_niche(niche, comments, emb, filmable):
    in_niche = (comments["niche"] == niche) & (comments["is_need"] == 1) & (comments["topic"] >= 0)
    train = in_niche & (comments["period"] == "train")
    X_train = unit(emb[train.values])
    centres, train_lab, k, guard = fit_angles(X_train, SEED_SETS[0])
    ari, per_angle = stability(X_train, train_lab)

    rows = np.where(in_niche.values)[0]
    sims = unit(emb[rows]) @ centres.T
    local = np.where(sims.max(1) >= guard, sims.argmax(1), -1)

    # ids ordered by size: 10xx AI Career, 20xx AI Tools, 30xx AI Agents
    order = np.argsort(-np.bincount(train_lab[train_lab >= 0], minlength=len(centres)))
    new_id = {int(old): NICHE_OFFSET[niche] * 10 + i for i, old in enumerate(order)}
    new_id[-1] = -1
    stab = {new_id[o]: v for o, v in per_angle.items()}
    c = comments.iloc[rows].assign(subtopic=[new_id[int(x)] for x in local], sub_sim=sims.max(1))
    is_train = (c["period"] == "train").values
    assigned = {p: float((c.loc[c["period"] == p, "subtopic"] >= 0).mean()) for p in ("train", "holdout")}
    words = top_keywords(c.loc[is_train, "text_clean"].values, c.loc[is_train, "subtopic"].values, top_n=6)

    end = comments["published_at"].max()
    recent = c["published_at"] > end - pd.Timedelta(days=RECENT_DAYS)
    prof = []
    for sid, g in c[c["subtopic"] >= 0].groupby("subtopic"):
        shares = g.loc[g["period"] == "train", "topic"].value_counts(normalize=True)
        if shares.empty:
            shares = g["topic"].value_counts(normalize=True)
        parent = int(shares.index[0])
        also = [int(t) for t, s in shares.iloc[1:].items() if s >= ALSO_SHARE and filmable.get(t, 0) == 1]
        ex = g.sort_values("sub_sim", ascending=False)["text_clean"].drop_duplicates().head(2).tolist()
        prof.append({"niche": niche, "subtopic": sid, "topic": parent, "parent_share": float(shares.iloc[0]),
                     "also_under": "|".join(map(str, also)),
                     "filmable": int(filmable.get(parent, 0) == 1 or bool(also)),
                     "stability": round(stab.get(sid, np.nan), 2),
                     "n_users": g["author_hash"].nunique(),
                     "n_users_recent": g.loc[recent[g.index], "author_hash"].nunique(),
                     "n_users_holdout": g.loc[g["period"] == "holdout", "author_hash"].nunique(),
                     "keywords": words.get(sid, ""), "example_1": ex[0] if ex else "",
                     "example_2": ex[1] if len(ex) > 1 else ""})
    check = {"niche": niche, "train_questions": int(train.sum()), "k_means_k": k,
             "angles": len(centres), "assigned_train": assigned["train"],
             "assigned_holdout": assigned["holdout"], "stability_ari": ari,
             "stable_angles": int(sum(v >= STABLE for v in per_angle.values()))}
    return pd.DataFrame(prof), c[["comment_id", "subtopic", "sub_sim"]], check


def load():
    """Angle profiles with their names (hand names where given, keywords otherwise)."""
    s = pd.read_csv(OUT / "subtopics.csv", dtype={"also_under": str})
    s["also_under"] = s["also_under"].fillna("")
    s["subtopic_name"] = s["keywords"].str.split(", ").str[:3].str.join(" / ")
    s["named"] = False
    if NAMES.exists():
        n = pd.read_csv(NAMES).dropna(subset=["subtopic_name"]).set_index("subtopic")["subtopic_name"]
        s["subtopic_name"] = s["subtopic"].map(n).fillna(s["subtopic_name"])
        s["named"] = s["subtopic"].isin(n.index)
    return s


def for_topic(s, topic):
    """Filmable angles to show under one broad topic (its own, plus shared ones)."""
    under = (s["topic"] == topic) | s["also_under"].str.split("|").apply(lambda x: str(topic) in x)
    return s[under & (s["filmable"] == 1)].sort_values(["n_users_recent", "n_users"], ascending=False)


def run():
    comments = pd.read_csv(OUT / "comments_topics.csv", parse_dates=["published_at"])
    filmable = pd.read_csv(config.DATA_LABELS / "topic_names.csv").set_index("topic")["actionable"].to_dict()
    emb = load_comment_embeddings()
    profs, maps, checks = [], [], []
    for niche in config.NICHES:
        p, m, chk = run_niche(niche, comments, emb, filmable)
        print(f"{niche}: {chk['angles']} angles, median {int(p['n_users'].median())} people, "
              f"{chk['assigned_train']:.0%} of questions assigned, stability ARI {chk['stability_ari']:.2f}, "
              f"{chk['stable_angles']} stable angles")
        profs.append(p)
        maps.append(m)
        checks.append(chk)
    pd.concat(profs).to_csv(OUT / "subtopics.csv", index=False)
    pd.concat(maps).to_csv(OUT / "comments_subtopics.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT / "subtopics_check.csv", index=False)

    # angle-level ranking (same score as the broad topics) and H1 at angle level
    from src.backtest import DEV_CUTOFFS
    from src.scoring import Coverage, load_data
    comments, videos, _, videos_all = load_data()
    cov, end, s = Coverage(comments, videos_all), comments["published_at"].max(), load()
    ranked = pd.concat([rank_angles(comments, videos, s, n, end, cov) for n in config.NICHES], ignore_index=True)
    ranked.to_csv(OUT / "weekly_angles.csv", index=False)
    p = persistence(comments, s, DEV_CUTOFFS)
    p.to_csv(OUT / "angle_persistence.csv", index=False)
    print(f"Angles with enough evidence this week: {int(ranked['eligible'].sum())}; "
          f"H1 at angle level, mean Spearman {p['spearman'].mean():+.2f}")

    picks = pick_angles(ranked, pd.read_csv(OUT / "weekly_top3.csv"))
    picks.to_csv(OUT / "weekly_angle_picks.csv", index=False)
    for r in picks[picks["chosen"]].itertuples():
        print(f"{r.niche}: topic #{r.broad_rank} {r.topic_name} -> {r.angle_name} "
              f"({r.angle_users} people, evidence {r.angle_evidence})")


def pick_angles(ranked, top3):
    """Angles come from inside the broad Top 3, so both layers agree.

    For each broad Top 3 topic, its best angle (enough evidence first, then the
    opportunity score). The weekly pick is the highest-ranked broad topic whose
    best angle has enough evidence: the existing rule "a topic one video can
    answer, with evidence", now checked at angle level. If no angle in the niche
    has enough evidence, broad #1 is kept and its best angle is marked thin.
    """
    rows = []
    for niche, t3 in top3.sort_values("rank").groupby("niche", sort=False):
        a = ranked[ranked["niche"] == niche]
        for t in t3.itertuples():
            under = a[(a["parent"] == t.topic)
                      | a["also_under"].fillna("").astype(str).str.split("|").apply(lambda x: str(t.topic) in x)]
            under = under.sort_values(["eligible", "opportunity"], ascending=False)
            best = under.iloc[0] if len(under) else None
            rows.append({"niche": niche, "broad_rank": t.rank, "topic": t.topic, "topic_name": t.topic_name,
                         "subtopic": None if best is None else int(best["subtopic"]),
                         "angle_name": "" if best is None else best["topic_name"],
                         "angle_users": 0 if best is None else int(best["n_users"]),
                         "angle_uncovered": None if best is None else round(float(best["uncovered_share"]), 2),
                         "angle_evidence": "none" if best is None else best["confidence"],
                         "angle_ok": best is not None and bool(best["eligible"]),
                         "example": "" if best is None else best["example_1"]})
    p = pd.DataFrame(rows)
    p["chosen"] = False
    for niche, g in p.groupby("niche"):
        ok = g[g["angle_ok"]]
        p.loc[(ok if len(ok) else g).index[0], "chosen"] = True
    return p



# ---------------------------------------------------------------- angle ranking
def angle_frame(comments, s):
    """Comments relabelled with their angle, and an angle table shaped like topics,
    so the same scoring code (src/scoring.py) can rank angles."""
    m = pd.read_csv(OUT / "comments_subtopics.csv")
    # keep the original row index: Coverage looks comments up by it
    sub = comments["comment_id"].map(m.set_index("comment_id")["subtopic"])
    c = comments.assign(topic=sub)
    c = c[c["topic"].notna() & (c["topic"] >= 0)].astype({"topic": int})
    names = pd.read_csv(config.DATA_LABELS / "topic_names.csv").set_index("topic")
    t = s.rename(columns={"topic": "parent", "subtopic": "topic", "subtopic_name": "topic_name"}).assign(
        actionable=lambda d: ((d["filmable"] == 1) & (d["stability"] >= STABLE)).astype(int),
        video_format=lambda d: d["parent"].map(names["video_format"]))
    return c, t


def rank_angles(comments, videos, s, niche, end, cov):
    """Same opportunity score as the broad topics, computed among stable, filmable angles."""
    from src.scoring import score_topics
    c, t = angle_frame(comments, s)
    r = score_topics(c, videos.iloc[0:0], t, niche, end, cov)
    if r.empty:
        return r
    keep = ["parent", "also_under", "stability", "n_users_holdout", "example_1", "example_2"]
    return r.merge(t[["topic"] + keep], on="topic", how="left").rename(columns={"topic": "subtopic"})


def persistence(comments, s, cutoffs, days=28):
    """H1 at angle level: Spearman between askers in the 30 days before a cutoff and
    askers in the next `days` days, per niche and cutoff."""
    c, t = angle_frame(comments, s)
    out = []
    for cut in cutoffs:
        T = pd.Timestamp(cut, tz="UTC")
        for niche in config.NICHES:
            ids = t.loc[(t["niche"] == niche) & (t["actionable"] == 1), "topic"]
            cc = c[c["topic"].isin(ids)]
            before = cc[(cc["published_at"] > T - pd.Timedelta(days=30)) & (cc["published_at"] <= T)]
            after = cc[(cc["published_at"] > T) & (cc["published_at"] <= T + pd.Timedelta(days=days))]
            b = before.groupby("topic")["author_hash"].nunique().reindex(ids, fill_value=0)
            a = after.groupby("topic")["author_hash"].nunique().reindex(ids, fill_value=0)
            out.append({"cutoff": cut, "niche": niche, "spearman": b.corr(a, method="spearman")})
    return pd.DataFrame(out)


if __name__ == "__main__":
    run()
