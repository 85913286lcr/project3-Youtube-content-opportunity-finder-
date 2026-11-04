# 4. Exploratory Data Analysis (EDA) Findings

Slide 3. Three findings, and each one changed a later step.

| # | Finding | Evidence | How it changed the modelling |
|---|---|---|---|
| 1 | **Same question, new words** | "Is this really free?" vs "Do I have to pay for it?": TF-IDF finds **0 shared words**, but MiniLM meaning similarity is **0.54**. For **59%** of the 16,523 training comments, the closest comment in meaning shares almost no words. | **Group by meaning** (MiniLM sentence embeddings, L09), not by word counts |
| 2 | **No clear groups** | PCA (L05) on 6,170 AI Career comments: **one cloud**; silhouette only **0.02–0.09** | There is **no natural K**, so K is chosen for the business: each topic small enough for one video (section 6) |
| 3 | **One video fakes a trend** | **53%** of the training comments sit under just **10%** of the videos | **Count people, not comments**, and count at most **3 per video** in the Demand signal |
