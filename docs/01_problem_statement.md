# 1. Problem Statement

Slide 1.

## What business problem are you solving, and who is affected?
I am an **AI YouTube creator** with three content lines: **AI Career, AI Tools and AI Agents**. My viewers leave thousands of comments, and many of them ask for videos I could make. But there are **too many comments** to read, and the same question comes in many different words. So it is hard to see what people really want, and I risk filming topics nobody asked for or that are already well covered.

The people affected are the creator (time and wrong topics) and the viewers (their questions stay unanswered).

## What decision or action will the model support?
**Which 3 videos to make each week**, one per content line, backed by real viewer questions. The tool suggests; a person makes the final choice.

My idea: **a video is worth making when the comments show 3 signals**:

| Signal | Meaning | Weight |
|---|---|---|
| Demand | many people ask it | 45% |
| Gap | few videos on it: asked, but no video yet | 35% |
| Freshness | rising this week | 20% |

Demand + Gap + Freshness = **Opportunity: worth filming now**.

## How is the problem handled today (the baseline)?
By hand: scrolling YouTube and reading comment sections, **about 3 hours a week**, then picking topics by gut feeling. For the machine-learning part, the baseline is a simple rule a person would use: **"a comment with a question mark is a question"** (section 6).
