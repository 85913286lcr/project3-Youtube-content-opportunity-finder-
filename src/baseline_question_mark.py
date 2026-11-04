"""Baseline used in the slides: a comment is a question if it contains "?".

Reads only data/labels/need_labels.csv (500 sampled; 493 after the Hindi-in-English filter:
356 train, 137 October test).

    python -m src.baseline_question_mark
"""
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score

from src import config
from src.preprocess import is_hinglish


def main():
    lab = pd.read_csv(config.DATA_LABELS / "need_labels.csv")
    lab = lab[~lab["text_clean"].map(is_hinglish)]      # same filter as cleaning: 500 -> 493
    for period, name in (("train", "train (356)"), ("holdout", "October test (137)")):
        d = lab[lab["period"] == period]
        y, pred = d["is_need"].values, d["text_clean"].str.contains("?", regex=False).astype(int).values
        print(f"{name:20s} kept {pred.sum():3d}  right {(pred & y).sum():3d}  "
              f"precision {precision_score(y, pred):.2f}  recall {recall_score(y, pred):.2f}  "
              f"F0.5 {fbeta_score(y, pred, beta=0.5):.2f}")


if __name__ == "__main__":
    main()
