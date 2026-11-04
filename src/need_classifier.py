"""Audience-need detector: is a comment a question, a request for content, or a
stated problem the viewer wants help with?

Labels: data/labels/need_labels.csv (500 hand-labelled comments)
    - 360 from the train period  -> model selection with 5-fold CV (356 after cleaning)
    - 140 from the holdout period -> sealed test set, used once (137 after cleaning)

Usage:
    python -m src.need_classifier
"""
import json
import warnings
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, fbeta_score,
                             make_scorer, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src import config

warnings.filterwarnings("ignore", category=FutureWarning)
from src.embed import load_comment_embeddings

MODEL_PATH = config.DATA_PROCESSED / "need_model.joblib"
RESULTS_PATH = config.DATA_PROCESSED / "need_model_results.json"

QUESTION_START = re.compile(
    r"^(how|what|which|why|where|when|who|can|could|should|would|is|are|do|does|did|will|any)\b",
    re.I)
REQUEST = re.compile(
    r"\b(?:please|pls|plz|make a video|do a video|video on|tutorial on|can you|could you|"
    r"need help|help me|stuck|not working|error|confused|any suggestions?|advice)\b", re.I)

# precision matters more than recall here: a false positive adds noise to the
# demand count of a topic, a missed comment only makes the count a bit smaller
F05 = make_scorer(fbeta_score, beta=0.5)


def rule_features(texts):
    texts = pd.Series(texts).astype(str)
    return np.c_[
        texts.str.contains(r"\?").astype(int),
        texts.str.match(QUESTION_START).astype(int),
        texts.str.contains(REQUEST).astype(int),
    ]


class RuleBaseline(BaseEstimator, ClassifierMixin):
    """Needs a question mark or a request phrase. This is roughly what a person
    skimming comments would do with Ctrl+F."""

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        f = rule_features(X)
        p = ((f[:, 0] == 1) | (f[:, 2] == 1)).astype(float)
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def load_labels():
    lab = pd.read_csv(config.DATA_LABELS / "need_labels.csv")
    comments = pd.read_csv(config.DATA_PROCESSED / "comments_clean.csv", usecols=["comment_id"])
    idx = pd.Series(np.arange(len(comments)), index=comments["comment_id"])
    lab["row"] = lab["comment_id"].map(idx)
    # a few labelled comments were removed by later cleaning steps
    return lab.dropna(subset=["row"]).astype({"row": int})


def feature_sets(lab, emb):
    X_emb = emb[lab["row"].values]
    X_rule = rule_features(lab["text_clean"])
    return {
        "text": lab["text_clean"].to_numpy(dtype=object),
        "emb": X_emb,
        "emb_rule": np.c_[X_emb, X_rule],
    }


def candidate_models():
    """(name, feature set, estimator, param grid). Model families from L03, L04 and L07."""
    return [
        ("Majority class (Dummy)", "text", DummyClassifier(strategy="most_frequent"), {}),
        ("Rule baseline", "text", RuleBaseline(), {}),
        ("TF-IDF + LogReg", "text",
         make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                       LogisticRegression(max_iter=2000, class_weight="balanced")),
         {"logisticregression__C": [0.1, 0.3, 1, 3, 10]}),
        ("MiniLM + LogReg", "emb",
         LogisticRegression(max_iter=2000, class_weight="balanced"),
         {"C": [0.03, 0.1, 0.3, 1, 3, 10]}),
        ("MiniLM + rules + LogReg", "emb_rule",
         LogisticRegression(max_iter=2000, class_weight="balanced"),
         {"C": [0.03, 0.1, 0.3, 1, 3, 10]}),
        ("MiniLM + rules + Random Forest", "emb_rule",
         RandomForestClassifier(class_weight="balanced", random_state=config.RANDOM_STATE, n_jobs=-1),
         {"n_estimators": [300], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 3, 5]}),
        # L07: a small neural network on the same features. No class_weight option in
        # sklearn's MLP, so the threshold (chosen on out-of-fold predictions) handles imbalance.
        ("MiniLM + rules + MLP (neural network)", "emb_rule",
         make_pipeline(StandardScaler(), MLPClassifier(max_iter=2000, random_state=config.RANDOM_STATE)),
         {"mlpclassifier__hidden_layer_sizes": [(32,), (64,), (64, 32)],
          "mlpclassifier__alpha": [0.001, 0.01, 0.1, 1.0, 3.0]}),
        ("MiniLM + rules + HistGradientBoosting", "emb_rule",
         HistGradientBoostingClassifier(class_weight="balanced", random_state=config.RANDOM_STATE),
         {"learning_rate": [0.03, 0.1], "max_depth": [3, 6], "max_iter": [200]}),
    ]


def scores(y, pred, proba):
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "f0.5": fbeta_score(y, pred, beta=0.5, zero_division=0),
        "pr_auc": average_precision_score(y, proba),
    }


def pick_threshold(y, proba, min_precision=0.75):
    """Lowest threshold on out-of-fold probabilities that still gives the target
    precision. Falls back to the F0.5-optimal threshold."""
    grid = np.linspace(0.2, 0.9, 71)
    ok = [t for t in grid if precision_score(y, proba >= t, zero_division=0) >= min_precision
          and (proba >= t).sum() >= 10]
    if ok:
        return float(min(ok))
    f = [fbeta_score(y, proba >= t, beta=0.5, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f))])


def run():
    emb = load_comment_embeddings()
    lab = load_labels()
    train = lab[lab["period"] == "train"].reset_index(drop=True)
    test = lab[lab["period"] == "holdout"].reset_index(drop=True)
    F_tr, F_te = feature_sets(train, emb), feature_sets(test, emb)
    y_tr, y_te = train["is_need"].values, test["is_need"].values
    cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)

    rows, fitted, tuning = [], {}, {}
    for name, fs, est, grid in candidate_models():
        if grid:
            gs = GridSearchCV(est, grid, scoring=F05, cv=cv, n_jobs=-1)
            gs.fit(F_tr[fs], y_tr)
            best = gs.best_estimator_
            tuning[name] = {"best_params": gs.best_params_,
                            "cv_results": pd.DataFrame(gs.cv_results_)[
                                ["params", "mean_test_score", "std_test_score"]
                            ].astype({"params": str}).to_dict("records")}
        else:
            best = est.fit(F_tr[fs], y_tr)
        oof = cross_val_predict(best, F_tr[fs], y_tr, cv=cv, method="predict_proba")[:, 1]
        thr = 0.5 if name in ("Rule baseline", "Majority class (Dummy)") else pick_threshold(y_tr, oof)
        r = {"model": name, "threshold": thr,
             **{f"cv_{k}": v for k, v in scores(y_tr, oof >= thr, oof).items()}}
        rows.append(r)
        fitted[name] = (best, fs, thr)

    table = pd.DataFrame(rows).sort_values("cv_f0.5", ascending=False)
    best_name = table.iloc[0]["model"]
    best, fs, thr = fitted[best_name]
    best.fit(F_tr[fs], y_tr)

    # the test set is touched once, for the selected model and the baseline only
    test_rows = []
    for name in [best_name, "Rule baseline"]:
        m, f, t = fitted[name]
        p = m.predict_proba(F_te[f])[:, 1]
        test_rows.append({"model": name, **scores(y_te, p >= t, p)})
    test_table = pd.DataFrame(test_rows)

    # error analysis on the test set
    p_te = best.predict_proba(F_te[fs])[:, 1]
    test = test.assign(proba=p_te, pred=(p_te >= thr).astype(int))
    errors = test[test["pred"] != test["is_need"]][["text_clean", "is_need", "proba"]]
    errors.to_csv(config.DATA_PROCESSED / "need_test_errors.csv", index=False)

    # score every comment with the selected model
    comments = pd.read_csv(config.DATA_PROCESSED / "comments_clean.csv")
    all_f = {"text": comments["text_clean"].to_numpy(dtype=object), "emb": emb,
             "emb_rule": np.c_[emb, rule_features(comments["text_clean"])]}
    comments["need_proba"] = best.predict_proba(all_f[fs])[:, 1]
    comments["is_need"] = (comments["need_proba"] >= thr).astype(int)
    comments.to_csv(config.DATA_PROCESSED / "comments_scored.csv", index=False)

    joblib.dump({"model": best, "features": fs, "threshold": thr, "name": best_name}, MODEL_PATH)
    table.to_csv(config.DATA_PROCESSED / "need_model_cv.csv", index=False)
    test_table.to_csv(config.DATA_PROCESSED / "need_model_test.csv", index=False)
    with open(RESULTS_PATH, "w") as f:
        json.dump({"selected": best_name, "threshold": thr, "tuning": tuning}, f, indent=1, default=str)

    pd.set_option("display.width", 160)
    print(table.round(3).to_string(index=False))
    print("\nselected:", best_name, "threshold", round(thr, 2))
    print(test_table.round(3).to_string(index=False))
    print("\nshare of comments flagged as need:",
          comments.groupby("period")["is_need"].mean().round(3).to_dict())


if __name__ == "__main__":
    run()
