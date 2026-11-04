# 7. Hyperparameter Tuning and Findings

Slide 6.

## Tuning method
- **Grid search with 5-fold stratified cross-validation** (L04), scored by F0.5.
- Run on the **356 training labels only**.

Gradient boosting, learning rate × tree depth:

| | depth 3 | depth 6 |
|---|---|---|
| learning rate 0.03 | 0.676 | 0.630 |
| learning rate 0.1 | 0.677 | **0.684** |

Kept: learning rate 0.1, depth 6. **Default 0.65 → tuned 0.68: only +0.03**, smaller than the CV noise (SD 0.10).

## Which hyperparameter had the biggest effect, and why
**Leaf size (`min_samples_leaf`)** had the biggest effect. Learning rate and depth only moved the score between 0.63 and 0.68.

| Leaf size | 1 | 5 | **20 (default)** | 40 |
|---|---|---|---|---|
| CV F0.5 | 0.57 | 0.59 | **0.68** | 0.65 |

On its own training data the model scores 1.00 at every leaf size.

Why leaf size matters so much:
- there are only 356 rows but 387 features;
- with tiny leaves, the trees memorise single comments, so they score perfectly on training data and badly on new comments;
- a leaf of 20 forces each leaf to cover a group of comments, so the model learns patterns.

With this little data, **more labels help more than more tuning**.

## Test set sealed
The 137 October labels were not used for any choice: features, model, tuning or threshold. They were opened once, for the final evaluation.
