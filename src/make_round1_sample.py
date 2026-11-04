"""Round 1 label sample (rebuilt): which 500 comments were labelled, and how they were picked.

The original sampling step was not saved. This script rebuilds it and checks that it
gives exactly the same 500 comments as data/labels/need_labels.csv.

How the 500 were picked:
    pool   = cleaned comments, before the Hindi-in-English filter was added
             (that filter came later and removed 7 of the 500)
    train  = 120 random comments per niche from 28 Jun - 27 Sep  -> 360
    test   = 140 random comments from 28 Sep - 26 Oct            -> 140
    seed   = random_state 42

It never touches need_labels.csv. It writes data/labels/round1/round1_sample_rebuilt.csv
and refuses to overwrite it.

Usage:
    python -m src.make_round1_sample
"""
import pandas as pd

from src import config
from src import preprocess as pp

N_PER_NICHE_TRAIN, N_TEST, SEED = 120, 140, config.RANDOM_STATE
LABELS = config.DATA_LABELS / "need_labels.csv"
OUT = config.DATA_LABELS / "round1" / "round1_sample_rebuilt.csv"


def pool():
    """Same cleaning as src/preprocess.py, without the later Hindi-in-English step."""
    videos = pd.read_csv(config.DATA_PROCESSED / "videos.csv")
    comments = pd.read_csv(config.DATA_PROCESSED / "comments_raw.csv")
    lang = videos["default_audio_language"].fillna("en").str.lower()
    english_title = videos["title"].map(pp.clean_text).map(pp.is_english)
    videos = videos[lang.str.startswith("en") & english_title]

    c = comments[comments["video_id"].isin(videos["video_id"])].copy()        # English videos
    c["text_clean"] = c["text"].map(pp.clean_text)
    c = c[c["text_clean"].str.split().str.len().fillna(0) >= 5]               # at least 5 words
    c = c[c["text_clean"].map(pp.is_english)]                                 # English comments
    dup = c["text_clean"].str.lower().map(c["text_clean"].str.lower().value_counts())
    c = c[dup < 3]                                                            # copy-paste spam
    c["period"] = pp.split_period(pd.to_datetime(c["published_at"], utc=True))
    return c.merge(videos[["video_id", "niche"]], on="video_id")


def main():
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite: {OUT}")
    c = pool()
    train = (c[c["period"] == "train"]
             .groupby("niche", group_keys=False)
             .apply(lambda g: g.sample(N_PER_NICHE_TRAIN, random_state=SEED)))
    test = c[c["period"] == "holdout"].sample(N_TEST, random_state=SEED)
    sample = pd.concat([train, test])[["comment_id", "period", "niche", "text_clean"]]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUT, index=False)

    labelled = set(pd.read_csv(LABELS)["comment_id"])
    same = sample["comment_id"].isin(labelled).sum()
    print(f"pool: {len(c):,} comments")
    print(f"sample: {len(train)} train (120 per niche) + {len(test)} test = {len(sample)}")
    print(f"same as need_labels.csv: {same} of {len(labelled)}")


if __name__ == "__main__":
    main()
