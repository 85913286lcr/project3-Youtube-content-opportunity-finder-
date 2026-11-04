"""Step 2, Prioritise: turn topics into ranked content opportunities.

For one niche, at a given date T, over the last 30 days:

    Audience Demand (D)   distinct people asking about the topic. Each video
                          contributes at most 3 questions (its most liked), so
                          one viral video cannot dominate a topic.
    Coverage Gap (G)      "answered a little": u = share of those questions with
                          no closely related video published in the 90 days
                          before T (best title cosine < 0.60; the video the
                          question was asked under does not count). G is highest
                          when about 70% are unanswered (u = GAP_PEAK) and falls
                          towards both ends: u near 0 = already well covered,
                          u = 1 = nobody has made it, which in the backtest
                          usually meant few people want it (H2, docs/08).
    Freshness (F)         daily rate of askers in the last 7 days vs the 23 days
                          before, smoothed: (last7 + 1) / ((prev23 + 1) * 7/23)

    Each is turned into a 0-100 percentile rank inside the niche, then
        Opportunity = 0.45 D + 0.35 G + 0.20 F
    The weights are an initial product assumption, not learned.

Evidence confidence
    High    >= 15 distinct askers, >= 5 source videos, >= 3 channels
    Medium  >= 8 distinct askers, >= 3 source videos
    Low     otherwise. Low topics never enter the Top 3, they go to a watch list.
"""
import numpy as np
import pandas as pd

from src import config
from src.embed import load_comment_embeddings, load_title_embeddings

OUT = config.DATA_PROCESSED
WEIGHTS = {"demand": 0.45, "gap": 0.35, "freshness": 0.20}
WINDOW_DAYS, RECENT_DAYS, COVER_LOOKBACK_DAYS = 30, 7, 90
GAP_PEAK = 0.70                 # chosen on the 7 development dates; see docs/08
COVER_THRESHOLD = 0.60          # from data/labels/coverage_pairs_labelled.csv
PER_VIDEO_CAP = 3


# ---------------------------------------------------------------- data
def load_data():
    comments = pd.read_csv(OUT / "comments_topics.csv", parse_dates=["published_at"])
    videos_all = pd.read_csv(OUT / "videos_clean.csv", parse_dates=["published_at"])
    matches = pd.read_csv(OUT / "video_topic_matches.csv")
    names = pd.read_csv(config.DATA_LABELS / "topic_names.csv")
    topics = pd.read_csv(OUT / "topics.csv").merge(names.drop(columns="niche"), on="topic", how="left")
    if topics["topic_name"].isna().any():
        raise ValueError("topic_names.csv does not match topics.csv, re-label after re-clustering")
    comments = comments.merge(videos_all[["video_id", "channel_id"]], on="video_id", how="left")
    videos = videos_all.merge(matches, on="video_id", how="inner")   # one row per (video, topic)
    return comments, videos, topics, videos_all


class Coverage:
    """Cosine between every need comment and every video title, computed once, so
    coverage can be checked at any date."""

    def __init__(self, comments, videos_all):
        E_c, E_t = load_comment_embeddings(), load_title_embeddings()
        need = np.where(comments["is_need"].values == 1)[0]
        self.row_of = pd.Series(np.arange(len(need)), index=comments.index[need])
        self.S = E_c[need] @ E_t.T
        self.v_date = videos_all["published_at"].dt.tz_convert(None).values
        self.v_id = videos_all["video_id"].to_numpy(dtype=object)

    def best_match(self, rows, own_video_ids, end, lookback=COVER_LOOKBACK_DAYS):
        """Best title cosine for each comment among videos published in
        (end - lookback, end], excluding the video the comment is on."""
        end = pd.Timestamp(end).tz_convert(None).to_datetime64()
        ok = (self.v_date <= end) & (self.v_date > end - np.timedelta64(lookback, "D"))
        if len(rows) == 0 or not ok.any():
            return np.full(len(rows), -1.0), np.array([None] * len(rows), dtype=object)
        S = self.S[self.row_of.loc[rows].values][:, ok]
        vid = self.v_id[ok]
        S = np.where(vid[None, :] == np.asarray(own_video_ids, dtype=object)[:, None], -1.0, S)
        j = S.argmax(1)
        return S[np.arange(len(rows)), j], vid[j]


def pct_rank(s):
    if len(s) <= 1:
        return pd.Series(50.0, index=s.index)
    return (s.rank(method="average") - 1) / (len(s) - 1) * 100


def confidence(n_users, n_videos, n_channels):
    if n_users >= 15 and n_videos >= 5 and n_channels >= 3:
        return "High"
    if n_users >= 8 and n_videos >= 3:
        return "Medium"
    return "Low"


def capped_questions(c):
    """At most PER_VIDEO_CAP questions per (topic, video), most liked first."""
    c = c.sort_values("like_count", ascending=False, kind="stable")
    return c.groupby(["topic", "video_id"], group_keys=False).head(PER_VIDEO_CAP)


# ---------------------------------------------------------------- scoring
def topic_signals(comments, topics, niche, end, coverage, exclude_video=None):
    """Raw D/G/F signals and evidence counts per topic at date `end`.
    `exclude_video` drops one video's comments (used for supervised training rows,
    so a video's own comments never describe the demand before it)."""
    end = pd.Timestamp(end)
    start = end - pd.Timedelta(days=WINDOW_DAYS)
    recent = end - pd.Timedelta(days=RECENT_DAYS)
    tp = topics[(topics["niche"] == niche) & (topics["actionable"] == 1)]
    c = comments[(comments["niche"] == niche) & (comments["topic"].isin(tp["topic"]))
                 & (comments["published_at"] > start) & (comments["published_at"] <= end)]
    if exclude_video is not None:
        c = c[c["video_id"] != exclude_video]
    if c.empty:
        return pd.DataFrame()
    c = capped_questions(c)
    best, _ = coverage.best_match(c.index, c["video_id"].values, end)
    c = c.assign(covered=best >= COVER_THRESHOLD)

    g = c.groupby("topic")
    d = pd.DataFrame({
        "n_questions": g.size(),
        "n_users": g["author_hash"].nunique(),
        "n_source_videos": g["video_id"].nunique(),
        "n_channels": g["channel_id"].nunique(),
        "likes": g["like_count"].sum(),
        "uncovered_share": 1 - g["covered"].mean(),
        "users_last7": c[c["published_at"] > recent].groupby("topic")["author_hash"].nunique(),
        "users_prev23": c[c["published_at"] <= recent].groupby("topic")["author_hash"].nunique(),
    })
    d[["users_last7", "users_prev23"]] = d[["users_last7", "users_prev23"]].fillna(0)
    d["freshness"] = (d["users_last7"] + 1) / (
        (d["users_prev23"] + 1) * RECENT_DAYS / (WINDOW_DAYS - RECENT_DAYS))
    return d.reset_index()


def score_topics(comments, videos, topics, niche, end, coverage, weights=WEIGHTS):
    d = topic_signals(comments, topics, niche, end, coverage)
    if d.empty:
        return d
    end = pd.Timestamp(end)
    v = videos[(videos["topic"].isin(d["topic"])) & (videos["published_at"] > end - pd.Timedelta(days=WINDOW_DAYS))
               & (videos["published_at"] <= end)]
    d["videos_on_topic_30d"] = d["topic"].map(v.groupby("topic").size()).fillna(0).astype(int)
    d = d.merge(topics[["topic", "topic_name", "video_format", "keywords"]], on="topic", how="left")
    d["D"] = pct_rank(d["n_users"]).round(0)
    d["G"] = pct_rank(-(d["uncovered_share"] - GAP_PEAK).abs()).round(0)
    d["F"] = pct_rank(d["freshness"]).round(0)
    d["opportunity"] = (weights["demand"] * d["D"] + weights["gap"] * d["G"]
                        + weights["freshness"] * d["F"]).round(1)
    d["confidence"] = [confidence(u, v_, ch) for u, v_, ch in
                       zip(d["n_users"], d["n_source_videos"], d["n_channels"])]
    d["eligible"] = d["confidence"] != "Low"
    d["niche"], d["end"] = niche, end
    return d.sort_values(["eligible", "opportunity"], ascending=[False, False]).reset_index(drop=True)


def evidence(comments, videos, topic, niche, end, coverage, n_comments=8, n_videos=6):
    """The raw material behind a recommendation: questions from distinct people
    with their source video, and the videos already on the topic."""
    end = pd.Timestamp(end)
    start = end - pd.Timedelta(days=WINDOW_DAYS)
    c = comments[(comments["topic"] == topic) & (comments["niche"] == niche)
                 & (comments["published_at"] > start) & (comments["published_at"] <= end)]
    c = capped_questions(c).drop_duplicates("author_hash")
    best, best_vid = coverage.best_match(c.index, c["video_id"].values, end)
    c = c.assign(cover_sim=best, cover_video=best_vid, covered=best >= COVER_THRESHOLD)
    c = c.sort_values(["like_count", "need_proba"], ascending=False)
    v = videos[(videos["topic"] == topic) & (videos["published_at"] <= end)
               & (videos["published_at"] > end - pd.Timedelta(days=COVER_LOOKBACK_DAYS))]
    v = v.sort_values("view_count", ascending=False)
    return (c[["text_clean", "like_count", "video_id", "published_at", "covered", "cover_video"]].head(n_comments),
            v[["title_clean", "channel_title", "view_count", "published_at", "video_id"]].head(n_videos))


def performance_residuals(videos_one_row_each):
    """Over-performance of each video: residual of a linear regression (L03) of
    log views on log subscribers and log age. 0 = as expected for its channel."""
    from sklearn.linear_model import LinearRegression

    v = videos_one_row_each
    X = np.c_[np.log1p(v["subscriber_count"].fillna(0)), np.log(v["age_days"])]
    y = np.log1p(v["view_count"])
    lr = LinearRegression().fit(X, y)
    return pd.Series((y - lr.predict(X)).to_numpy(), index=v["video_id"].values), lr


def topic_reach(videos, end, days=90, min_videos=3):
    """Evidence only. How existing videos on each topic did against expectation,
    as a multiple of expected views (1.0 = as expected)."""
    end = pd.Timestamp(end)
    v = videos[(videos["published_at"] > end - pd.Timedelta(days=days))
               & (videos["published_at"] <= end)].copy()
    resid, _ = performance_residuals(v.drop_duplicates("video_id"))
    v["residual"] = v["video_id"].map(resid)
    g = v.groupby("topic")["residual"].agg(["median", "size"])
    g = g[g["size"] >= min_videos]
    return np.exp(g["median"]).rename("reach_multiple")


if __name__ == "__main__":
    comments, videos, topics, videos_all = load_data()
    cov = Coverage(comments, videos_all)
    end = comments["published_at"].max()
    pd.set_option("display.width", 250)
    for niche in config.NICHES:
        s = score_topics(comments, videos, topics, niche, end, cov)
        print(f"\n=== {niche} (30 days to {end.date()}) ===")
        print(s[["topic", "topic_name", "n_users", "n_source_videos", "uncovered_share", "freshness",
                 "D", "G", "F", "opportunity", "confidence"]].head(10).round(2).to_string(index=False))
