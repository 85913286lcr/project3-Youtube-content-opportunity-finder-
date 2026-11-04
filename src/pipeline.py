"""Run every step in order.

    python -m src.pipeline            # everything except data collection
    python -m src.pipeline --collect  # also pull fresh data from the YouTube API
"""
import sys

from src import (backtest, clustering, collect_youtube, embed, monitoring, need_classifier,
                 preprocess, recommend, subtopics, supply)

if __name__ == "__main__":
    if "--collect" in sys.argv:
        collect_youtube.main()
    for step in (preprocess, embed, need_classifier, clustering, supply, backtest,
                 monitoring, recommend, subtopics):
        print(f"\n===== {step.__name__} =====")
        step.run()
