"""Pull videos, channel stats and top-level comments from the YouTube Data API v3.

Usage:
    python -m src.collect_youtube

Quota (default 10,000 units/day):
    search.list        100 units per call  -> 15 queries x 2 pages = 3,000
    videos.list          1 unit per 50 ids
    channels.list        1 unit per 50 ids
    commentThreads.list  1 unit per page   -> ~600 videos x 3 pages = ~1,800

The script is resumable. Raw responses are appended to JSONL files and videos
that already have comments are skipped, so if the quota runs out you can run it
again the next day.
"""
import json
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src import config

VIDEOS_JSONL = config.DATA_RAW / "videos.jsonl"
COMMENTS_JSONL = config.DATA_RAW / "comments.jsonl"
CHANNELS_JSONL = config.DATA_RAW / "channels.jsonl"
SEARCH_JSONL = config.DATA_RAW / "search_hits.jsonl"


def get_client():
    load_dotenv(config.ROOT / ".env")
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        raise SystemExit("YOUTUBE_API_KEY is missing. Put it in the .env file.")
    return build("youtube", "v3", developerKey=key, cache_discovery=False)


def read_jsonl(path):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, rows):
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def search_videos(yt):
    done = {(r["niche"], r["query"]) for r in read_jsonl(SEARCH_JSONL)}
    published_after = (datetime.now(timezone.utc) - timedelta(days=config.LOOKBACK_DAYS))
    published_after = published_after.strftime("%Y-%m-%dT%H:%M:%SZ")

    for niche, queries in config.NICHES.items():
        for q in queries:
            if (niche, q) in done:
                continue
            hits, token = [], None
            for _ in range(config.SEARCH_PAGES_PER_QUERY):
                resp = yt.search().list(
                    q=q, part="id", type="video", maxResults=50,
                    publishedAfter=published_after, relevanceLanguage="en",
                    regionCode="US", order="relevance", pageToken=token,
                ).execute()
                hits += [it["id"]["videoId"] for it in resp.get("items", [])]
                token = resp.get("nextPageToken")
                if not token:
                    break
            append_jsonl(SEARCH_JSONL, [{"niche": niche, "query": q, "rank": i, "video_id": v}
                                        for i, v in enumerate(hits)])
            print(f"[search] {niche:10s} | {q:35s} -> {len(hits)} videos")


def fetch_video_details(yt):
    hits = read_jsonl(SEARCH_JSONL)
    have = {r["id"] for r in read_jsonl(VIDEOS_JSONL)}
    ids = sorted({h["video_id"] for h in hits} - have)
    for i in range(0, len(ids), 50):
        resp = yt.videos().list(part="snippet,statistics,contentDetails",
                                id=",".join(ids[i:i + 50])).execute()
        append_jsonl(VIDEOS_JSONL, resp.get("items", []))
    print(f"[videos] fetched details for {len(ids)} new videos")


def fetch_channel_details(yt):
    videos = read_jsonl(VIDEOS_JSONL)
    have = {r["id"] for r in read_jsonl(CHANNELS_JSONL)}
    ids = sorted({v["snippet"]["channelId"] for v in videos} - have)
    for i in range(0, len(ids), 50):
        resp = yt.channels().list(part="statistics", id=",".join(ids[i:i + 50])).execute()
        append_jsonl(CHANNELS_JSONL, resp.get("items", []))
    print(f"[channels] fetched {len(ids)} new channels")


def fetch_comments(yt):
    from src.collect_authors import author_hash
    salt = os.getenv("AUTHOR_SALT", "")
    videos = read_jsonl(VIDEOS_JSONL)
    done = {r["video_id"] for r in read_jsonl(COMMENTS_JSONL)}
    todo = [v for v in videos
            if v["id"] not in done and int(v["statistics"].get("commentCount", 0) or 0) > 0]
    print(f"[comments] {len(todo)} videos to fetch")

    for n, v in enumerate(todo, 1):
        vid, rows, token = v["id"], [], None
        try:
            for _ in range(config.COMMENT_PAGES_PER_VIDEO):
                resp = yt.commentThreads().list(
                    part="snippet", videoId=vid, maxResults=100,
                    order="relevance", textFormat="plainText", pageToken=token,
                ).execute()
                for it in resp.get("items", []):
                    s = it["snippet"]["topLevelComment"]["snippet"]
                    # author name is not kept. The author channel id is only stored
                    # as a salted hash, enough to count distinct people (privacy)
                    rows.append({
                        "author_hash": author_hash(
                            (s.get("authorChannelId") or {}).get("value"), salt),
                        "comment_id": it["id"],
                        "video_id": vid,
                        "text": s.get("textDisplay", ""),
                        "like_count": s.get("likeCount", 0),
                        "reply_count": it["snippet"].get("totalReplyCount", 0),
                        "published_at": s.get("publishedAt"),
                    })
                token = resp.get("nextPageToken")
                if not token:
                    break
        except HttpError as e:
            reason = str(e)
            if "quotaExceeded" in reason:
                print("Quota exceeded. Run the script again tomorrow to continue.")
                return
            # comments disabled or video removed: mark as done with no rows
            rows = []
        if not rows:
            rows = [{"comment_id": None, "video_id": vid, "text": None}]
        append_jsonl(COMMENTS_JSONL, rows)
        if n % 25 == 0:
            print(f"  {n}/{len(todo)} videos")
        time.sleep(0.05)


def build_tables():
    """Flatten the raw JSONL files into the three CSVs used everywhere else."""
    hits = pd.DataFrame(read_jsonl(SEARCH_JSONL))
    # first niche/query that found the video, by niche order then rank
    niche_order = {n: i for i, n in enumerate(config.NICHES)}
    hits["niche_order"] = hits["niche"].map(niche_order)
    first = hits.sort_values(["niche_order", "rank"]).drop_duplicates("video_id")
    n_niches = hits.groupby("video_id")["niche"].nunique().rename("n_niches")

    vrows = []
    for v in read_jsonl(VIDEOS_JSONL):
        sn, st = v["snippet"], v.get("statistics", {})
        vrows.append({
            "video_id": v["id"],
            "title": sn.get("title"),
            "description": (sn.get("description") or "")[:500],
            "channel_id": sn.get("channelId"),
            "channel_title": sn.get("channelTitle"),
            "published_at": sn.get("publishedAt"),
            "duration": v.get("contentDetails", {}).get("duration"),
            "default_audio_language": sn.get("defaultAudioLanguage"),
            "view_count": int(st.get("viewCount", 0) or 0),
            "like_count": int(st.get("likeCount", 0) or 0),
            "comment_count": int(st.get("commentCount", 0) or 0),
        })
    videos = pd.DataFrame(vrows).drop_duplicates("video_id")
    videos = videos.merge(first[["video_id", "niche", "query"]], on="video_id", how="left")
    videos = videos.merge(n_niches, on="video_id", how="left")

    ch = pd.DataFrame([{"channel_id": c["id"],
                        "subscriber_count": int(c["statistics"].get("subscriberCount", 0) or 0)}
                       for c in read_jsonl(CHANNELS_JSONL)]).drop_duplicates("channel_id")
    videos = videos.merge(ch, on="channel_id", how="left")

    comments = pd.DataFrame(read_jsonl(COMMENTS_JSONL))
    comments = comments.dropna(subset=["comment_id"]).drop_duplicates("comment_id")
    authors = config.DATA_PROCESSED / "comment_authors.csv"
    if "author_hash" not in comments.columns and authors.exists():
        comments = comments.merge(pd.read_csv(authors), on="comment_id", how="left")

    videos.to_csv(config.DATA_PROCESSED / "videos.csv", index=False)
    comments.to_csv(config.DATA_PROCESSED / "comments_raw.csv", index=False)
    print(f"[tables] videos={len(videos)}  comments={len(comments)}")
    return videos, comments


def main():
    yt = get_client()
    search_videos(yt)
    fetch_video_details(yt)
    fetch_channel_details(yt)
    fetch_comments(yt)
    build_tables()


if __name__ == "__main__":
    main()
