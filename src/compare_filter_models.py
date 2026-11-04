"""Extra numbers for slide "Compare models & tune" (question filter).

TRAINING labels only. Same 5-fold StratifiedKFold, same features (MiniLM + 3 rules),
same threshold rule and the tuned settings from need_classifier.py / need_model_results.json.
Does not change any existing file; writes data/processed/filter_model_compare.csv.

Usage:
    python -m src.compare_filter_models
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src import config
from src.embed import load_comment_embeddings
from src.need_classifier import (RuleBaseline, feature_sets, load_labels, pick_threshold,
                                 rule_features, scores)

warnings.filterwarnings("ignore")
OUT = config.DATA_PROCESSED / "filter_model_compare.csv"
RS = config.RANDOM_STATE

MODELS = {  # tuned settings, as selected by GridSearchCV in need_classifier.py
    "Logistic regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=1),
    "Random forest": RandomForestClassifier(class_weight="balanced", random_state=RS, n_jobs=1,
                                            n_estimators=300, max_depth=None, min_samples_leaf=3),
    "Gradient boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=RS,
                                                        learning_rate=0.1, max_depth=6, max_iter=200),
}


def evaluate(est, X, y, cv):
    oof = cross_val_predict(est, X, y, cv=cv, method="predict_proba")[:, 1]
    thr = pick_threshold(y, oof)
    cv_s = scores(y, oof >= thr, oof)
    fitted = clone(est).fit(X, y)
    tr = fitted.predict_proba(X)[:, 1]
    tr_s = scores(y, tr >= thr, tr)
    return oof, thr, cv_s, tr_s


def main():
    lab = load_labels()
    train = lab[lab["period"] == "train"].reset_index(drop=True)
    y = train["is_need"].values
    F = feature_sets(train, load_comment_embeddings())
    X = F["emb_rule"]
    cv = StratifiedKFold(5, shuffle=True, random_state=RS)
    rows, oofs = [], {}

    # baseline as coded: "?" OR request phrase
    f = rule_features(train["text_clean"])
    for name, pred in [("Rules (as coded: ? or request phrase)", (f[:, 0] | f[:, 2]).astype(int)),
                       ("Rules (? or question word or request)", (f[:, 0] | f[:, 1] | f[:, 2]).astype(int))]:
        s = scores(y, pred, pred)
        rows.append({"model": name, "cv_f0.5": s["f0.5"], "precision": s["precision"], "recall": s["recall"],
                     "train_f0.5": s["f0.5"], "gap": 0.0, "threshold": None})
    rule_pred = (f[:, 0] | f[:, 2]).astype(int)

    for name, est in MODELS.items():
        oof, thr, cv_s, tr_s = evaluate(est, X, y, cv)
        oofs[name] = (oof, thr)
        rows.append({"model": name, "cv_f0.5": cv_s["f0.5"], "precision": cv_s["precision"],
                     "recall": cv_s["recall"], "train_f0.5": tr_s["f0.5"],
                     "gap": tr_s["f0.5"] - cv_s["f0.5"], "threshold": thr})

    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)
    pd.set_option("display.width", 160)
    print(f"training labels: {len(y)}, needs: {y.sum()}\n")
    print(table.round(2).to_string(index=False))

    # gradient boosting: its own min_samples_leaf effect (other settings as tuned)
    print("\nGradient boosting, min_samples_leaf (CV F0.5 / train F0.5):")
    for leaf in [1, 5, 20, 40]:
        est = clone(MODELS["Gradient boosting"]).set_params(min_samples_leaf=leaf)
        _, _, cv_s, tr_s = evaluate(est, X, y, cv)
        print(f"  leaf {leaf:>2}: CV {cv_s['f0.5']:.2f}  train {tr_s['f0.5']:.2f}")

    # needs that are not questions: rules missed, gradient boosting (out-of-fold) caught
    oof, thr = oofs["Gradient boosting"]
    t = train["text_clean"].astype(str)
    mask = (y == 1) & (rule_pred == 0) & (oof >= thr) & ~t.str.contains(r"\?")
    print(f"\nNeeds without '?', missed by rules, caught by gradient boosting (out-of-fold): {mask.sum()}")
    for txt, p in zip(t[mask], oof[mask]):
        print(f"  p={p:.2f} | {txt}")


if __name__ == "__main__":
    main()
