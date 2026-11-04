"""Which videos are about which topic: semantic search (L09) with the topic name
as the query. Used to show "videos already on this topic" and how they performed
against their channel's expected views.

Coverage of individual questions (the Coverage Gap score) is a different step,
see question_coverage() in src/scoring.py.

Each actionable topic has a human-written name that reads like a search query
("MCP explained: what it adds on top of an API"). We embed it with the same
all-MiniLM-L6-v2 model, rank every video title by cosine similarity, and count a
video as covering the topic when cosine >= 0.65. A video can cover more than one
topic.

Why 0.65: in L09 cosine is a ranking signal, not a percentage, so the cut-off has
to come from labelled data. data/labels/supply_pairs_labelled.csv has 190
hand-checked (topic, title) pairs (top 4 titles plus rank 12 for each of the 38
actionable topics). Above 0.65, 71% of pairs are on topic.

An earlier version matched titles to the comment-cluster centroids instead. It
did not work well: viewers ask questions in a different style from how titles
are written, and broad topics pulled in almost every title.

Usage:
    python -m src.supply
"""
import numpy as np
import pandas as pd

from src import config
from src.embed import encode, load_title_embeddings

OUT = config.DATA_PROCESSED
COS_THRESHOLD = 0.65


def precision_report(labels_path=config.DATA_LABELS / "supply_pairs_labelled.csv"):
    d = pd.read_csv(labels_path)
    rows = []
    for t in [0.50, 0.55, 0.60, 0.65, 0.70]:
        s = d[d["cosine"] >= t]
        rows.append({"threshold": t, "pairs_above": len(s), "precision": s["relevant"].mean(),
                     "relevant_missed": int(d[d["cosine"] < t]["relevant"].sum())})
    at5 = d[d["rank"] <= 4].groupby("topic_name")["relevant"].mean().sort_values()
    return pd.DataFrame(rows), at5


def run():
    videos = pd.read_csv(OUT / "videos_clean.csv")
    names = pd.read_csv(config.DATA_LABELS / "topic_names.csv")
    names = names[names["actionable"] == 1].reset_index(drop=True)
    E = load_title_embeddings()
    N = encode(names["topic_name"].tolist())
    S = E @ N.T

    vi, ti = np.where(S >= COS_THRESHOLD)
    matches = pd.DataFrame({"video_id": videos["video_id"].values[vi],
                            "topic": names["topic"].values[ti], "cosine": S[vi, ti]})
    matches.to_csv(OUT / "video_topic_matches.csv", index=False)
    print(f"matches={len(matches)}  videos covering >=1 topic="
          f"{matches.video_id.nunique()}/{len(videos)}")
    print(matches.groupby("topic").size().reindex(names["topic"]).fillna(0)
          .astype(int).to_string())


if __name__ == "__main__":
    run()
