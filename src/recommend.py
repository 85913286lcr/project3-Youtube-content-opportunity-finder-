"""Weekly run: Top 3 per niche, a watch list, and the evidence behind each pick.

Ranking: Opportunity = 0.45 D + 0.35 G + 0.20 F (percentile ranks inside the niche).
Comments do not tell us views, so the creator makes the final pick and the A/B test
(most-asked vs opportunity picks) checks whether unanswered topics get more views.
Only High / Medium evidence topics can enter the Top 3. Low evidence topics with
a high opportunity go to the watch list.

Writes
    data/processed/weekly_top3.csv
    data/processed/weekly_watchlist.csv
    reports/weekly_briefs.md

Usage:
    python -m src.recommend
"""
import pandas as pd

from src import config
from src.scoring import Coverage, evidence, load_data, score_topics, topic_reach

OUT = config.DATA_PROCESSED
REPORTS = config.ROOT / "reports"


def rank_niche(comments, videos, topics, niche, end, cov, reach):
    s = score_topics(comments, videos, topics, niche, end, cov)
    if s.empty:
        return s
    s["priority"] = s["opportunity"].round(1)
    s["reach_multiple"] = s["topic"].map(reach).round(2)
    return s.sort_values(["eligible", "priority"], ascending=[False, False]).reset_index(drop=True)


def run():
    comments, videos, topics, videos_all = load_data()
    cov = Coverage(comments, videos_all)
    end = comments["published_at"].max()
    reach = topic_reach(videos, end)

    tops, watch = [], []
    md = [f"# Weekly content opportunities\n\nData up to {end.date()}. Demand window 30 days.\n"]
    for niche in config.NICHES:
        s = rank_niche(comments, videos, topics, niche, end, cov, reach)
        top = s[s["eligible"]].head(3).assign(rank=lambda d: range(1, len(d) + 1))
        wl = s[~s["eligible"]].sort_values("opportunity", ascending=False).head(3)
        tops.append(top)
        watch.append(wl)
        md.append(f"\n## {niche}\n")
        for r in top.itertuples():
            ev_c, ev_v = evidence(comments, videos, r.topic, niche, end, cov)
            md.append(f"\n### {r.rank}. {r.topic_name}\n")
            md.append(
                f"- Format: {r.video_format}. Evidence: {r.confidence}\n"
                f"- {int(r.n_users)} different people asked in the last 30 days, under "
                f"{int(r.n_source_videos)} videos from {int(r.n_channels)} channels\n"
                f"- {r.uncovered_share:.0%} of their questions have no closely matching recent video\n"
                f"- Askers per day, last 7 days vs the 23 before: {r.freshness:.1f}x\n"
                f"- Demand {r.D:.0f}, Gap {r.G:.0f}, Freshness {r.F:.0f} -> opportunity {r.opportunity:.0f}\n"
                + (f"- Existing videos on this topic got {r.reach_multiple:.1f}x expected views\n"
                   if pd.notna(r.reach_multiple) else "- Too few existing videos to judge their reach\n"))
            md.append("\nWhat viewers asked:\n")
            for q in ev_c.itertuples():
                tag = "" if not q.covered else " (a related video exists)"
                md.append(f"- \"{q.text_clean[:220]}\" ({q.like_count} likes){tag} "
                          f"https://www.youtube.com/watch?v={q.video_id}\n")
            md.append("\nVideos already on this topic:\n" if len(ev_v) else
                      "\nVideos already on this topic: none in the last 90 days.\n")
            for v in ev_v.itertuples():
                md.append(f"- {v.title_clean} ({v.channel_title}, {v.view_count:,} views)\n")
        if len(wl):
            md.append("\n**Watch list** (low evidence so far): "
                      + "; ".join(f"{r.topic_name} ({int(r.n_users)} askers)" for r in wl.itertuples()) + "\n")

    cols = ["niche", "rank", "topic", "topic_name", "video_format", "confidence", "n_users",
            "n_source_videos", "n_channels", "uncovered_share", "freshness", "D", "G", "F",
            "opportunity", "priority", "reach_multiple"]
    top = pd.concat(tops, ignore_index=True)[cols]
    top.to_csv(OUT / "weekly_top3.csv", index=False)
    pd.concat(watch, ignore_index=True).to_csv(OUT / "weekly_watchlist.csv", index=False)
    (REPORTS / "weekly_briefs.md").write_text("".join(md))
    pd.set_option("display.width", 220)
    print(top[["niche", "rank", "topic_name", "confidence", "n_users", "uncovered_share",
               "opportunity", "reach_multiple"]].round(2).to_string(index=False))


if __name__ == "__main__":
    run()
