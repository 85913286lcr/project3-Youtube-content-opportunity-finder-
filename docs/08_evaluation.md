# 8. Evaluation Metrics

Slides 5, 7, 9 and 1.

## Which metrics and why
- **Only 21% are questions**, so **accuracy misleads**: a model that says "no" to everything is already 79% right but finds nothing.
- **Precision:** are the kept comments real questions? **Recall:** did we find all real ones?
- **Fake questions inflate a topic** and send the creator the wrong way, so **precision comes first**:
  - **F0.5** ranks models (precision counts two times);
  - a business floor to pass: **precision ≥ 75%** (3 in 4 kept are real). The threshold is the lowest one that meets it.
- **Topics:** stability (ARI), and how many new questions still fit a topic.
- **Weekly picks:** back-test hit rate.

## Validation and held-out test performance
**Validation: 5-fold stratified CV** (each fold ≈ 71 comments, 15 questions).
- Gradient boosting's 5 scores: 0.55, 0.59, 0.71, 0.77, 0.80, average **0.68 ± 0.10 (SD)**.
- Only 3 of 5 beat the baseline, and the gap 0.09 is smaller than the SD, so CV alone could not tell.

**Held-out test: 137 October comments, opened once, 19 real questions.**

| | Caught | False alarms | Precision | Recall | F0.5 |
|---|---|---|---|---|---|
| **Gradient boosting** | 13 of 19 | 5 | 72% | 68% | **0.71** |
| "?" rule | 10 of 19 | 13 | 43% | 53% | 0.45 |

- The gap on the test is **0.26**, much bigger than the CV noise, so gradient boosting is really better.
- There is no drop from CV (0.68) to test (0.71), so it works on new comments.
- Precision is 72%, just below the 75% goal. With only 19 test questions, this number is uncertain.

**Why so few labels were enough.**
- **Pretrained MiniLM** already knows meaning (trained on over 1B sentence pairs); the labels only teach the last step (transfer learning).
- **The rules are strong:** "?" is in 54% of questions but only 9% of other comments.

The final model reads all 25,226 comments and keeps **3,678 questions**: 2,574 Jul–Sep + 1,104 October.

**Topics on new data.**
- Each topic has a centre and a ring that holds 90% of its Jul–Sep questions (cosine distance, L09).
- October questions inside a ring: **AI Career 88%, AI Tools 78%, AI Agents 81%**.
- So the topics are still usable; the rest are new themes, and the topics are re-clustered monthly.

**Weekly picks: back-test (L06).**
- On **7 past Mondays (3 Aug – 14 Sep 2026)**, the Top 3 were picked with older data only, then checked over the next 4 weeks.
- **7 in 10 picks** (44 of 63) were still asked a lot (top half of their content line), vs **under 5 in 10** (47%) for random picks.
- A final check on 28 Sep gave 67% vs 49%.

## Error analysis: where does the model fail, and for which segments?
On the 137 October comments there were 11 errors: **6 missed questions and 5 false alarms**.

- **Missed questions have no "?"**:
  - "For freshers can I get placed by mastering these things";
  - "How to go viral on YouTube";
  - "oh, I cannot find GitHub MCP server in Extensions".
- **False alarms are rhetorical questions**:
  - "How little do you value the job to have candidates talk to a robot??";
  - "Task app with priority sorting … actually worked first try?"
- **By segment:** **AI Career** has the most errors (7 of 11). Its comments are long personal stories that mix advice-seeking and opinion. AI Tools and AI Agents have 1 and 3.

**Fix:** label more questions written without "?" and more rhetorical "?" comments (next labelling round).
