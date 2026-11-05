# 6. Models Tried

Slides 5, 6, 8 and 9.

## Step 1: find questions (supervised)
- **Baseline:** a simple rule, "a comment with **?** is a question" (`src/baseline_question_mark.py`).
- Then 3 models of increasing complexity, all on the best features (meaning + 3 rules) with balanced class weights.
- 5-fold stratified cross-validation on the 356 training labels.

| Model | Precision | Recall | CV F0.5 | vs baseline |
|---|---|---|---|---|
| Baseline: "?" only | 0.61 | 0.54 | 0.59 | – |
| Logistic regression (L03) | 0.75 | 0.41 | 0.64 | +0.05 |
| Random forest (L04) | 0.79 | 0.30 | 0.59 | ±0 |
| **Gradient boosting (L04)** | 0.75 | 0.49 | **0.68** | **+0.09** |

Gradient boosting was best by CV. But its 5 fold scores swing ±0.10, so the gap is not proven by CV alone; the final test decides (section 8).

## Step 2: group into topics (unsupervised)
Four ways to group the **2,574 Jul–Sep questions**, compared by stability.

**ARI (Adjusted Rand Index, L05):** run each method 5 times with different random seeds and compare every pair of runs (average ARI). 1 = identical groups, 0 = no better than random.

| Method | AI Career | AI Tools | AI Agents |
|---|---|---|---|
| By words: TF-IDF + K-Means (baseline) | 0.30 | 0.23 | 0.43 |
| By meaning: MiniLM + K-Means (L09) | 0.37 | 0.47 | **0.58** |
| By meaning + PCA: 384 → ~90 numbers, 80% of info kept (L05) | **0.49** | **0.53** | 0.49 |
| HDBSCAN* (drops 15–32% of questions) | 0.34 | 0.51 | 0.21 |

Grouping by meaning was the steadiest in all 3 content lines, so it is used.
- By words, one AI Tools group mixed a paid course, GitHub links and a mountain biking accident.
- By meaning, the matching group was all "Can I do the same with Claude?", which one video can answer.

**K-Means per content line: choosing K** (slide 9).
- K = how many topics each content line is cut into; each topic becomes one video idea.
- K = 8, 10, 12, 15, 18, 20, 25 were tried.
- **The data shows no natural K:** the elbow curve is a straight line (no bend), and silhouette is only 0.03–0.09.
- So K is chosen with 2 rules (my choice):
  1. **1 video can't cover too much:** the biggest topic holds at most 12% of the line's questions (about 2× an average topic; AI Career: at most 109 of 912);
  2. **not too many:** 10–25 topics, so a creator can read the list.
- Among the K that pass both rules, the highest silhouette wins (AI Career: 15 and 20 almost tie, 0.0438 vs 0.0437).

| Content line | Questions | K |
|---|---|---|
| AI Career | 912 | **15** |
| AI Tools | 684 | **20** |
| AI Agents | 978 | **15** |
| **Total** | 2,574 | **50 topics** |

\* beyond the course
