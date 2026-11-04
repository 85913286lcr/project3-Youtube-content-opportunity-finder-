"""Add a hashed author id to every comment we already have.

The first collection did not keep author ids. To count how many *different*
people ask about a topic we need them, but we do not need to know who they are.
So each author channel id is hashed with a secret salt (HMAC-SHA256, kept in
.env) right after it is fetched, and only the hash is stored.

The comments are re-requested by their own id (commentThreads.list?id=...), so
the comment set stays exactly the same as before. Cost: 1 quota unit per 50
comments.

Usage:
    python -m src.collect_authors
"""
import hashlib
import hmac
import os

import pandas as pd
from dotenv import load_dotenv

from src import config
from src.collect_youtube import get_client

OUT = config.DATA_PROCESSED / "comment_authors.csv"


def author_hash(channel_id, salt):
    if not channel_id:
        return None
    return hmac.new(salt.encode(), channel_id.encode(), hashlib.sha256).hexdigest()[:16]


def run():
    load_dotenv(config.ROOT / ".env")
    salt = os.getenv("AUTHOR_SALT")
    if not salt:
        raise SystemExit("AUTHOR_SALT is missing in .env")
    yt = get_client()

    ids = pd.read_csv(config.DATA_PROCESSED / "comments_raw.csv", usecols=["comment_id"])["comment_id"]
    done = set(pd.read_csv(OUT)["comment_id"]) if OUT.exists() else set()
    todo = [i for i in ids if i not in done]
    print(f"{len(todo)} comments to look up")

    rows = []
    for n in range(0, len(todo), 50):
        batch = todo[n:n + 50]
        resp = yt.commentThreads().list(part="snippet", id=",".join(batch),
                                        textFormat="plainText").execute()
        found = {}
        for it in resp.get("items", []):
            s = it["snippet"]["topLevelComment"]["snippet"]
            cid = (s.get("authorChannelId") or {}).get("value")
            found[it["id"]] = author_hash(cid, salt)
        # comments deleted since the first collection come back missing
        rows += [{"comment_id": c, "author_hash": found.get(c)} for c in batch]
        if len(rows) >= 2500 or n + 50 >= len(todo):
            pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            rows = []
            print(f"  {min(n + 50, len(todo))}/{len(todo)}")

    a = pd.read_csv(OUT)
    print(f"done: {len(a)} comments, {a.author_hash.notna().mean():.1%} with an author, "
          f"{a.author_hash.nunique()} distinct authors")


if __name__ == "__main__":
    run()
