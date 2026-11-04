"""Topic discovery: cluster the audience-need comments of each niche separately.

Follows the L05 recipe (PCA -> K-Means -> pick K -> check stability -> profile
and name), applied to L09 sentence embeddings, and compares it with a density
method that goes beyond the course.

Candidates (per niche, train period only)
    Baseline. TF-IDF + K-Means                        (bag of words)
    Model A.  MiniLM + K-Means, PCA on/off as a tuning choice
    Model B.  MiniLM + UMAP + HDBSCAN                 (beyond the course, see
              docs/course_alignment.md). HDBSCAN can leave comments out
              ("noise") instead of forcing every comment into a topic.

All candidates are scored in the same space (MiniLM, cosine) so the numbers are
comparable. Topic ids are niche-prefixed: 1xx AI Career, 2xx AI Tools, 3xx AI Agents.
New comments are assigned to the nearest topic centroid *of their own niche*, or
left unassigned if they are not close enough to any topic.

Usage:
    python -m src.clustering
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score

from src import config
from src.embed import load_comment_embeddings

warnings.filterwarnings("ignore")

RS = config.RANDOM_STATE
OUT = config.DATA_PROCESSED
K_GRID = [8, 10, 12, 15, 18, 20, 25]
HDB_GRID = [(nn, mcs) for nn in (10, 15, 30) for mcs in (8, 12, 16, 20)]
PCA_VARIANCE = 0.80
NICHE_OFFSET = {n: 100 * (i + 1) for i, n in enumerate(config.NICHES)}

EXTRA_STOP = {"video", "videos", "thanks", "thank", "sir", "bro", "please", "pls", "make",
              "like", "just", "know", "want", "need", "use", "using", "does", "did", "don",
              "really", "good", "great", "hi", "hello", "guys", "ve", "ll", "im", "can", "get",
              "one", "would", "could", "also", "much", "way", "think", "doesn", "isn"}

METHOD_NAMES = {
    "tfidf_kmeans": "Baseline. TF-IDF + K-Means",
    "kmeans": "Model A. MiniLM + K-Means",
    "pca_kmeans": "Model A. MiniLM + PCA + K-Means",
    "umap_hdbscan": "Model B. MiniLM + UMAP + HDBSCAN",
}


def niche_slug(niche):
    return niche.lower().replace(" ", "_")


# ---------------------------------------------------------------- fitting
def fit_labels(method, X_emb, texts, params, seed=RS, n_pca=None):
    """Cluster labels for one configuration (-1 = left out, HDBSCAN only)."""
    if method == "tfidf_kmeans":
        Xt = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.5,
                             sublinear_tf=True).fit_transform(texts)
        return KMeans(params["k"], n_init=10, random_state=seed).fit_predict(Xt)
    if method == "kmeans":
        return KMeans(params["k"], n_init=10, random_state=seed).fit_predict(X_emb)
    if method == "pca_kmeans":
        Z = PCA(n_pca, random_state=RS).fit_transform(X_emb)
        return KMeans(params["k"], n_init=10, random_state=seed).fit_predict(Z)
    if method == "umap_hdbscan":
        from umap import UMAP
        Z = UMAP(n_neighbors=params["n_neighbors"], n_components=5, min_dist=0.0,
                 metric="cosine", random_state=seed).fit_transform(X_emb)
        return HDBSCAN(min_cluster_size=params["min_cluster_size"], min_samples=3).fit_predict(Z)
    raise ValueError(method)


def evaluate(X_emb, labels):
    """Internal metrics in MiniLM space. Unassigned points (-1) are excluded."""
    mask = labels >= 0
    k = len(set(labels[mask]))
    out = {"n_clusters": k, "unassigned_pct": float(1 - mask.mean()),
           "max_share": float(pd.Series(labels[mask]).value_counts(normalize=True).max())}
    if k < 2:
        return {**out, "silhouette": np.nan, "davies_bouldin": np.nan, "coherence": np.nan}
    out["silhouette"] = float(silhouette_score(X_emb[mask], labels[mask], metric="cosine"))
    out["davies_bouldin"] = float(davies_bouldin_score(X_emb[mask], labels[mask]))
    sims = []
    for c in set(labels[mask]):
        m = X_emb[labels == c]
        cen = m.mean(0) / np.linalg.norm(m.mean(0))
        sims.append((m @ cen).mean())
    out["coherence"] = float(np.mean(sims))   # mean cosine of a comment to its own centroid
    return out


def inertia_curve(Z, ks):
    return [KMeans(k, n_init=10, random_state=RS).fit(Z).inertia_ for k in ks]


def seed_stability(method, X_emb, texts, params, n_pca, seeds=(42, 123, 7, 2024, 99)):
    """L05 extension 1: refit with different random seeds, compare labels with ARI.
    For HDBSCAN the seed changes the UMAP layout."""
    labs = [fit_labels(method, X_emb, texts, params, seed=s, n_pca=n_pca) for s in seeds]
    return float(np.mean([adjusted_rand_score(labs[i], labs[j])
                          for i in range(len(labs)) for j in range(i + 1, len(labs))]))


def is_usable(r):
    """Business constraints per niche. Silhouette is low for every setting, so
    like K = 4 vs 6 in L05 the choice is partly about actionability: a topic
    must be narrow enough to be one video (no topic above 12% of the niche's
    questions), 10 to 25 topics per niche is what a creator can read, and at
    most 35% of questions may be left out of every topic."""
    return (10 <= r["n_clusters"] <= 25 and r["max_share"] <= 0.12
            and r["unassigned_pct"] <= 0.35)


# ---------------------------------------------------------------- topics
class TopicModel:
    """Centroids in MiniLM space plus an assignment threshold."""

    def __init__(self, X_emb, labels, quantile=0.10):
        self.ids = np.array(sorted(set(labels[labels >= 0])))
        cents = np.stack([X_emb[labels == c].mean(0) for c in self.ids])
        self.centroids = cents / np.linalg.norm(cents, axis=1, keepdims=True)
        member_sims = np.concatenate([X_emb[labels == c] @ self.centroids[i]
                                      for i, c in enumerate(self.ids)])
        # a new comment must be at least as close as the 10% least typical members
        self.threshold = float(np.quantile(member_sims, quantile))

    def assign(self, X, threshold=None):
        thr = self.threshold if threshold is None else threshold
        sims = X @ self.centroids.T
        best = sims.argmax(1)
        best_sim = sims[np.arange(len(X)), best]
        return np.where(best_sim >= thr, self.ids[best], -1), best_sim

    def save(self, path):
        np.savez(path, ids=self.ids, centroids=self.centroids, threshold=self.threshold)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        obj = cls.__new__(cls)
        obj.ids, obj.centroids, obj.threshold = d["ids"], d["centroids"], float(d["threshold"])
        return obj


def load_topic_models():
    return {n: TopicModel.load(OUT / f"topic_model_{niche_slug(n)}.npz") for n in config.NICHES}


def assign_by_niche(models, emb, niches):
    """Assign each row to the topics of its own niche."""
    topic = np.full(len(emb), -1)
    sim = np.zeros(len(emb))
    niches = np.asarray(niches)
    for n, tm in models.items():
        m = niches == n
        if m.any():
            topic[m], sim[m] = tm.assign(emb[m])
    return topic, sim


def top_keywords(texts, labels, top_n=8):
    """Words that are frequent in one topic and rare in the others: TF-IDF where
    each topic's comments are joined into one document."""
    df = pd.DataFrame({"t": texts, "c": labels})
    df = df[df["c"] >= 0]
    docs = df.groupby("c")["t"].apply(" ".join)
    stop = list(CountVectorizer(stop_words="english").get_stop_words() | EXTRA_STOP)
    tf = TfidfVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\.\+#-]+\b")
    M = tf.fit_transform(docs).toarray()
    vocab = np.array(tf.get_feature_names_out())
    return {int(c): ", ".join(vocab[np.argsort(-M[i])[:top_n]]) for i, c in enumerate(docs.index)}


# ---------------------------------------------------------------- run
def grid_for(method):
    if method == "umap_hdbscan":
        return [{"n_neighbors": nn, "min_cluster_size": m} for nn, m in HDB_GRID]
    return [{"k": k} for k in K_GRID]


def run_niche(niche, comments, emb):
    need = (comments["is_need"] == 1) & (comments["niche"] == niche)
    tr = (need & (comments["period"] == "train")).values
    ho = (need & (comments["period"] == "holdout")).values
    X_tr, X_ho = emb[tr], emb[ho]
    T_tr = comments.loc[tr, "text_clean"].to_numpy(dtype=object)

    pca_full = PCA(random_state=RS).fit(X_tr)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_pca = int(np.searchsorted(cum, PCA_VARIANCE) + 1)
    pca_tab = pd.DataFrame({"niche": niche, "component": np.arange(1, len(cum) + 1),
                            "explained": pca_full.explained_variance_ratio_, "cumulative": cum})
    Z_tr = PCA(n_pca, random_state=RS).fit_transform(X_tr)
    elbow = pd.DataFrame({"niche": niche, "k": K_GRID, "inertia": inertia_curve(Z_tr, K_GRID)})

    rows = []
    for method in METHOD_NAMES:
        for params in grid_for(method):
            labels = fit_labels(method, X_tr, T_tr, params, n_pca=n_pca)
            rows.append({"niche": niche, "method": method, "params": json.dumps(params),
                         **evaluate(X_tr, labels)})
    grid = pd.DataFrame(rows)
    grid["usable"] = grid.apply(is_usable, axis=1)

    best = []
    for method in METHOD_NAMES:
        sub = grid[(grid["method"] == method) & grid["usable"]]
        if sub.empty:
            sub = grid[grid["method"] == method]
        r = sub.sort_values("silhouette", ascending=False).iloc[0].to_dict()
        r["stability_ari"] = seed_stability(method, X_tr, T_tr, json.loads(r["params"]), n_pca)
        r["model"] = METHOD_NAMES[method]
        r["usable"] = bool(is_usable(r))
        best.append(r)
    comp = pd.DataFrame(best)

    # choose among usable configs: best silhouette, ties (within 0.005) go to the more stable
    cand = comp[comp["usable"]] if comp["usable"].any() else comp
    top = cand[cand["silhouette"] >= cand["silhouette"].max() - 0.005]
    chosen = top.sort_values("stability_ari", ascending=False).iloc[0]
    method, params = chosen["method"], json.loads(chosen["params"])
    comp["chosen"] = comp["method"] == method

    labels_tr = fit_labels(method, X_tr, T_tr, params, n_pca=n_pca)
    # re-number topics by size and add the niche prefix (1xx, 2xx, 3xx)
    order = pd.Series(labels_tr[labels_tr >= 0]).value_counts().index
    remap = {old: NICHE_OFFSET[niche] + i for i, old in enumerate(order)}
    labels_tr = np.array([remap.get(l, -1) for l in labels_tr])
    tm = TopicModel(X_tr, labels_tr)
    tm.save(OUT / f"topic_model_{niche_slug(niche)}.npz")

    topic_tr = tm.assign(X_tr)[0]
    topic_ho = tm.assign(X_ho)[0]
    test = {"niche": niche, "method": method, "params": params, "n_pca": n_pca,
            "n_topics": int(len(tm.ids)), "assign_threshold": tm.threshold,
            "train": evaluate(X_tr, topic_tr), "holdout": evaluate(X_ho, topic_ho)}

    kw = top_keywords(T_tr, topic_tr)
    reps = {}
    for i, c in enumerate(tm.ids):
        m = np.where(topic_tr == c)[0]
        o = m[np.argsort(-(X_tr[m] @ tm.centroids[i]))]
        reps[int(c)] = list(T_tr[o[:6]])
    sub = comments.loc[tr].assign(topic=topic_tr)
    topics = (sub[sub["topic"] >= 0].groupby("topic")
              .agg(n_train=("comment_id", "size"), n_authors=("author_hash", "nunique"),
                   n_videos=("video_id", "nunique"), avg_likes=("like_count", "mean"))
              .reset_index())
    topics["niche"] = niche
    topics["keywords"] = topics["topic"].map(kw)
    topics["examples"] = topics["topic"].map(lambda c: " || ".join(reps[c]))
    return grid, comp, test, topics, pca_tab, elbow


def run():
    comments = pd.read_csv(OUT / "comments_scored.csv")
    emb = load_comment_embeddings()
    grids, comps, tests, topics, pcas, elbows = [], [], [], [], [], []
    for niche in config.NICHES:
        print(f"\n===== {niche} =====")
        g, c, t, tp, pv, el = run_niche(niche, comments, emb)
        grids.append(g); comps.append(c); tests.append(t); topics.append(tp)
        pcas.append(pv); elbows.append(el)
        print(c[["model", "params", "n_clusters", "unassigned_pct", "max_share", "silhouette",
                 "coherence", "stability_ari", "usable", "chosen"]].round(3).to_string(index=False))
        print("holdout:", {k: round(v, 3) for k, v in t["holdout"].items()})

    pd.concat(grids).to_csv(OUT / "clustering_grid.csv", index=False)
    pd.concat(comps).to_csv(OUT / "clustering_comparison.csv", index=False)
    pd.concat(topics).to_csv(OUT / "topics.csv", index=False)
    pd.concat(pcas).to_csv(OUT / "pca_variance.csv", index=False)
    pd.concat(elbows).to_csv(OUT / "elbow.csv", index=False)
    with open(OUT / "clustering_results.json", "w") as f:
        json.dump(tests, f, indent=1, default=str)

    models = load_topic_models()
    topic, sim = assign_by_niche(models, emb, comments["niche"].values)
    comments["topic"] = np.where(comments["is_need"] == 1, topic, -1)
    comments["topic_sim"] = sim
    comments.to_csv(OUT / "comments_topics.csv", index=False)
    old = OUT / "topic_model.npz"
    if old.exists():
        old.unlink()


if __name__ == "__main__":
    run()
