# 5. Feature Engineering

Slides 2 and 6.

## Cleaning, encoding, scaling, missing values and outliers
- **Cleaning:**
  - English videos and comments only;
  - at least 5 words;
  - Hindi written in English letters removed;
  - spam removed (same text 3+ times);
  - see section 3.
- **Missing values:**
  - 24 comments with no text were dropped;
  - a comment with no author hash counts as its own author.
- **Encoding text:**
  - **TF-IDF** word counts. Words like *how / what / can* are kept, although they are normally stop-words, because they are the signal of a question.
  - **all-MiniLM-L6-v2** sentence embeddings: 384 numbers per comment, unit length. This is the same pretrained model as in Lesson 9, used as is.
- **Scaling:** embeddings are already normalised; tree models need no scaling.
- **Outliers:** viral videos are the main outlier, so Demand counts at most 3 questions per video. Spam duplicates are removed.
- **Class imbalance:** models use `class_weight="balanced"`.

## New features and why
**3 rules: simple yes/no signs of a question.**
1. has a question mark;
2. starts with how / what / can;
3. says please / help.

MiniLM knows the topic but not the intent: a question and a compliment on the same topic look alike. The rules add the "is someone asking" signal.

| Features (same model: logistic regression) | CV F0.5 |
|---|---|
| Word counts (TF-IDF) | 0.61 |
| Meaning (MiniLM) | 0.45 |
| **Meaning + 3 rules (384 + 3 = 387 numbers)** | **0.64** |

For the score, each topic gets 3 new features per week (section 8):
- **Demand**: different people asking in the last 30 days;
- **Gap**: share of questions with no matching recent video;
- **Freshness**: last 7 days vs the 23 days before.

## How data leakage was avoided
- **Split by time (28 Sep):** nothing from October is used for training.
- **Test set locked:** the 137 October labels were opened once, after all choices were made.
- **Transformers fitted on training folds only:** TF-IDF sits inside the model pipeline, so in cross-validation it is refitted on each training fold.
- **Topics:** learned on Jul–Sep questions only, then checked on October.
- **Back-test:** each past Monday uses only data from before that day.
