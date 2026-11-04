# ▶️ Content Opportunity Finder

**Turns YouTube comments into 3 video ideas a week for AI creators.**
Congrui Liu · NTU Module 3 project · [slides](slides/) · [public dashboard](app/)

## Project overview and problem statement

**Problem.**
- I am an AI YouTube creator with three content lines: AI Career, AI Tools and AI Agents.
- Every week I must decide what to film next.
- Viewers leave thousands of comments; reading them by hand takes about **3 hours a week**, and the same question comes in many different words.

**What the tool does.** It reads the comments and suggests **3 video ideas a week**, backed by real viewer questions. My idea: **a video is worth making when the comments show 3 signals**:
- 👥 **Demand**: many people ask it;
- 📹 **Gap**: few videos on it;
- 🔥 **Freshness**: rising this week.

| 📥 Collect | 🧹 Clean | 🔍 Find questions | 🧩 Group into topics | 🏆 Score |
|---|---|---|---|---|
| 51,447 comments | 25,226 | 3,678 questions | 50 topics | 3 videos / week |
| YouTube API, 120 days | English, no spam | supervised (L03 · L04 · L09) | unsupervised K-Means (L05 · L09) | score + back-test (L06 · L09) |

The write-up follows the 11 presentation points of the project brief:

| # | Section | File | Slides |
|---|---|---|---|
| 1 | Problem statement | [docs/01_problem_statement.md](docs/01_problem_statement.md) | 1 |
| 2 | Type of ML problem | [docs/02_ml_problem_type.md](docs/02_ml_problem_type.md) | 1, 4, 8, 10 |
| 3 | Dataset | [docs/03_dataset.md](docs/03_dataset.md) | 2, 4 |
| 4 | EDA findings | [docs/04_eda.md](docs/04_eda.md) | 3 |
| 5 | Feature engineering | [docs/05_feature_engineering.md](docs/05_feature_engineering.md) | 2, 6 |
| 6 | Models tried | [docs/06_models.md](docs/06_models.md) | 5, 6, 8, 9 |
| 7 | Hyperparameter tuning | [docs/07_tuning.md](docs/07_tuning.md) | 6 |
| 8 | Evaluation metrics | [docs/08_evaluation.md](docs/08_evaluation.md) | 5, 7, 9, 1 |
| 9 | Opportunity sizing | [docs/09_opportunity_sizing.md](docs/09_opportunity_sizing.md) | 11 |
| 10 | Recommendation: deploy or not | [docs/10_recommendation.md](docs/10_recommendation.md) | 11 |
| 11 | Deployment plan | [docs/11_deployment_plan.md](docs/11_deployment_plan.md) | 11 |

## Dataset description and how to obtain it

- **Source:** official YouTube Data API v3. 3 niches × 5 search words; videos and top-level comments from the last 120 days (28 Jun – 26 Oct 2026).
- **Size:** 51,447 comments (6 columns) from 1,282 videos (15 columns) and 893 channels. After cleaning: 25,226.
- **Split by time:** train Jul–Sep (16,523), test October (8,703).
- **Target:** `is_need`, 1 = the comment asks for something a video could give. 493 labelled comments: 356 train (21% questions) and 137 test.
- **Privacy:** author names are never stored, only a salted hash used to count different people.

Details: [docs/03_dataset.md](docs/03_dataset.md).

**How to obtain it.** Raw comments are not in this repository (large, and they change every week). To collect your own:
1. Get a YouTube Data API key from Google Cloud.
2. Copy `.env.example` to `.env` and fill in `YOUTUBE_API_KEY` and `AUTHOR_SALT`.
3. Run `python -m src.pipeline --collect`. One run uses about 7,000 of the 10,000 free daily API units.

**Included in the repository:**

| Path | What it is |
|---|---|
| `data/labels/need_labels.csv` | the labelled comments (question 1 / not 0) |
| `data/labels/coverage_pairs_labelled.csv` | 150 checked comment–video title pairs, used to choose the 0.60 gap cut-off |
| `data/labels/topic_names.csv` | the 50 topic names |
| `results/` | topic-method comparison (ARI), back-test summary, this week's picks |
| `app/data/` | a small anonymised snapshot for the public dashboard |

## Instructions to set up the environment and reproduce the results

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.baseline_question_mark   # "?" baseline from the labels only: train F0.5 0.59, October test 0.45
python -m src.pipeline --collect       # collect data, then run every step
python -m src.pipeline                 # re-run every step on data already collected
python -m src.compare_filter_models    # side-by-side models on the training labels
```

`src.pipeline` runs, in order:
1. `preprocess`: clean the data and split it by time;
2. `embed`: MiniLM embeddings;
3. `need_classifier`: features, models, 5-fold CV, grid search, threshold, one final test;
4. `clustering`: 4 methods, ARI, K-Means topics;
5. `supply`: match recent videos to topics;
6. `backtest`: 7 past Mondays + final check;
7. `monitoring`: weekly topic fit;
8. `recommend`: weekly Top 3;
9. `subtopics`: one video angle per pick.

The random seed is fixed (`RANDOM_STATE = 42` in `src/config.py`).

**Dashboard:**

```bash
cd app && pip install -r requirements.txt && streamlit run app_public.py
```

## Summary of results

| Question | Result |
|---|---|
| Does the question filter beat a simple "?" rule on unseen October data? | Yes: gradient boosting F0.5 **0.71 vs 0.45**; caught 13 of 19 real questions vs 10 |
| Group by words or by meaning? | Meaning was the steadiest (ARI) in all 3 content lines; 15 + 20 + 15 = **50 topics** |
| Do the topics still fit new data? | **78–88%** of October questions fit the old topics |
| Are the weekly picks any good? | On 7 past Mondays, **7 in 10** picks were still hot a month later (random: under 5 in 10) |
| Value | Saves **2.5 h a week** (130 h a year); pays back in about 6 months |
| Recommendation | **Deploy now as a decision aid; roll out the ranking only if an 8-week A/B test wins** |
| Main limit | Too few labels (only 19 test questions); next step: label 500 more |

## Repository structure

```
Content Opportunity Finder/
├── README.md                 you are here
├── docs/                     sections 1–11 of the project brief (01_… to 11_…)
├── slides/                   final presentation (11 slides)
├── app/                      public Streamlit dashboard + anonymised data snapshot
├── src/
│   ├── config.py             niches, search words, paths, random seed
│   ├── collect_youtube.py    YouTube Data API collection (+ collect_authors.py: hashing)
│   ├── preprocess.py         cleaning (51,447 → 25,226) and the 28 Sep time split
│   ├── embed.py              all-MiniLM-L6-v2 sentence embeddings
│   ├── make_round1_sample.py how the labelled comments were sampled (seed 42)
│   ├── baseline_question_mark.py   the "?" baseline used in the slides
│   ├── need_classifier.py    features, models, 5-fold CV, grid search, threshold, test
│   ├── compare_filter_models.py    side-by-side model comparison on the training labels
│   ├── clustering.py         4 grouping methods, ARI stability, K-Means topics
│   ├── supply.py             matches recent videos to topics
│   ├── scoring.py            Demand · Gap · Freshness score (semantic search, cosine ≥ 0.60)
│   ├── backtest.py           7 past Mondays + final check
│   ├── monitoring.py         weekly topic fit (re-cluster below 75%)
│   ├── recommend.py          weekly Top 3 per content line
│   ├── subtopics.py          one video angle inside each pick
│   └── pipeline.py           runs every step in order
├── data/
│   ├── labels/               labelled data (see above)
│   ├── raw/, processed/      created by the pipeline (not committed)
├── results/                  key result tables
├── requirements.txt
└── .env.example              API key template (the real .env is never committed)
```
