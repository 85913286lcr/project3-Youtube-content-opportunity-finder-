"""Clean comments and videos and mark the time-based split.

Usage:
    python -m src.preprocess
"""
import html
import re

import numpy as np
import pandas as pd
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

from src import config

DetectorFactory.seed = 0

# Time split. Everything is fitted on the train period only.
# The holdout period is kept for testing, the backtest and the monitoring simulation.
TRAIN_END = pd.Timestamp("2026-09-28", tz="UTC")
DATA_END = pd.Timestamp("2026-10-26 23:59", tz="UTC")

URL_RE = re.compile(r"https?://\S+|www\.\S+")
TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\b")
MENTION_RE = re.compile(r"@[\w\-.]+")
EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF‍️]+"
)
WS_RE = re.compile(r"\s+")

# Romanised Hindi ("Hinglish") passes langdetect as English quite often. These
# words are very rare in English text, two or more of them is a reliable flag.
# Found after the first clustering run produced a topic made only of these words.
HINGLISH = {"hai", "hain", "bhai", "aap", "mujhe", "kya", "nahi", "nahin", "kaise", "karo",
            "kar", "bhi", "mera", "meri", "apne", "kuch", "banao", "banaye", "chahiye",
            "toh", "wala", "wali", "raha", "rahi", "hota", "hoga", "sab", "ek", "ye", "yeh",
            "ke", "ki", "ka", "ko", "se", "aur", "bhaiya", "didi", "sahi", "acha", "accha"}


def clean_text(t):
    t = html.unescape(str(t))
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = TIMESTAMP_RE.sub(" ", t)
    t = EMOJI_RE.sub(" ", t)
    return WS_RE.sub(" ", t).strip()


def is_english(t):
    # langdetect is unreliable on very short strings, so require some ASCII letters too
    letters = sum(ch.isalpha() for ch in t)
    if letters == 0 or sum(ch.isascii() for ch in t if ch.isalpha()) / letters < 0.9:
        return False
    try:
        return detect(t) == "en"
    except LangDetectException:
        return False


def is_hinglish(t):
    words = re.findall(r"[a-z]+", t.lower())
    return sum(w in HINGLISH for w in words) >= 2


def split_period(ts):
    return np.where(ts < TRAIN_END, "train", "holdout")


def run():
    videos = pd.read_csv(config.DATA_PROCESSED / "videos.csv", parse_dates=["published_at"])
    comments = pd.read_csv(config.DATA_PROCESSED / "comments_raw.csv", parse_dates=["published_at"])
    authors = config.DATA_PROCESSED / "comment_authors.csv"
    if "author_hash" not in comments.columns and authors.exists():
        comments = comments.merge(pd.read_csv(authors), on="comment_id", how="left")
    # a comment without an author hash counts as its own author
    comments["author_hash"] = comments["author_hash"].fillna(comments["comment_id"])
    n0 = len(comments)
    log = [("raw comments", n0)]

    # keep English-language videos only (audio language missing is allowed, title must be English)
    lang = videos["default_audio_language"].fillna("en").str.lower()
    videos["title_clean"] = videos["title"].map(clean_text)
    videos["title_en"] = videos["title_clean"].map(is_english)
    videos = videos[lang.str.startswith("en") & videos["title_en"]].copy()
    videos["period"] = split_period(videos["published_at"])
    age_days = (DATA_END - videos["published_at"]).dt.total_seconds() / 86400
    videos["age_days"] = age_days.clip(lower=1)
    videos["views_per_day"] = videos["view_count"] / videos["age_days"]
    # "outlier ratio": views relative to channel size, a common creator metric
    videos["views_per_sub"] = videos["view_count"] / videos["subscriber_count"].clip(lower=100)

    comments = comments[comments["video_id"].isin(videos["video_id"])].copy()
    log.append(("on English videos", len(comments)))

    comments["text_clean"] = comments["text"].map(clean_text)
    comments["n_words"] = comments["text_clean"].str.split().str.len().fillna(0).astype(int)
    comments = comments[comments["n_words"] >= 5]
    log.append(("at least 5 words", len(comments)))

    comments = comments[comments["text_clean"].map(is_english)]
    log.append(("detected as English", len(comments)))

    comments = comments[~comments["text_clean"].map(is_hinglish)]
    log.append(("romanised Hindi removed", len(comments)))

    # copy-paste spam: the same text posted 3+ times
    dup = comments["text_clean"].str.lower().map(comments["text_clean"].str.lower().value_counts())
    comments = comments[dup < 3]
    log.append(("spam removed", len(comments)))

    comments = comments.merge(videos[["video_id", "niche"]], on="video_id", how="left")
    comments["period"] = split_period(comments["published_at"])
    comments["like_count"] = comments["like_count"].fillna(0).astype(int)

    videos.to_csv(config.DATA_PROCESSED / "videos_clean.csv", index=False)
    comments.to_csv(config.DATA_PROCESSED / "comments_clean.csv", index=False)
    pd.DataFrame(log, columns=["step", "rows"]).to_csv(
        config.DATA_PROCESSED / "cleaning_log.csv", index=False)

    for step, n in log:
        print(f"{step:25s} {n:>7,}")
    print(comments.groupby(["period", "niche"]).size().unstack())
    print(videos.groupby(["period", "niche"]).size().unstack())


if __name__ == "__main__":
    run()
