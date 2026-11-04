# 2. Type of ML Problem

Slides 1, 4, 8 and 11. The project has two machine-learning steps and a scoring step.

| | Step 1: find questions | Step 2: group into topics | Step 3: score and pick |
|---|---|---|---|
| Supervised or unsupervised | **Supervised** (labelled comments) | **Unsupervised** (no answer key) | rule-based score, checked by a back-test |
| Classification or regression | **Classification** | clustering (K-Means) | ranking |
| Binary or multi-class | **Binary**: question (1) or not (0) | 50 topics | – |
| Why | I can label examples; a comment either asks for something or not | nobody tells the model what the topics are | there is no label for "a good video", so the weights are a product choice |

**Batch or real-time:** a **weekly batch**. Every Monday the tool reads new comments and suggests the Top 3.

**Time-series or not:** the models do not forecast a time series, but time order matters. All data is **split by time** (train Jul–Sep, test October), and the score is tested with a **rolling back-test on past Mondays** (L06).
