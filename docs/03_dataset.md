# 3. Dataset

Slides 2 and 4.

## Source, size and time period
- **Source:** official **YouTube Data API v3**. 3 niches (my 3 content lines) × 5 real search words, for example "how to get a job in AI". Videos and top-level comments.
- **Time period:** **120 days, 28 Jun – 26 Oct 2026**.
- **Size:**
  - comments: **51,447 rows × 6 columns** (comment id, video id, text, like count, reply count, publish time);
  - videos: **1,282 rows × 15 columns** (title, description, channel, publish time, views, likes, comment count, niche, search word, …);
  - **893 channels**.

**Cleaning: 51,447 → 25,226** (about half was noise).

| Step | Comments left |
|---|---|
| Collected | 51,447 |
| English videos only (langdetect*) | 40,805 |
| At least 5 words | 27,131 |
| English comments (langdetect*) | 25,658 |
| Hindi-in-English removed (word list*) | 25,424 |
| Spam removed (same text 3+ times) | 25,226 |

\* beyond the course

**Split by time (L06).**
- Cut on **28 Sep**.
- Train: Jul–Sep, **16,523** comments. Test: October, **8,703** comments.
- A random split would leak October comments into training. The tool only knows the past, so it is tested that way.

## Target variable and how it is defined
`is_need`:
- **1 = a question**: the viewer asks for something a video could give (a request, a problem, or a real question).
- **0** = everything else, for example praise: "Amazing video!".

**493 comments** were labelled:
- picked at random with a fixed seed (`src/make_round1_sample.py`), then split by time;
- **train 356** (74 questions, 21%) · **test 137** (19 questions, 14%);
- the labels follow a written guide.

## Known limitations
- **Class imbalance:** only 21% of comments are questions.
- **Few labels:** 356 to train, only 19 test questions, so test numbers have a wide range.
- **Bias:**
  - English only, 3 niches;
  - YouTube search favours big channels;
  - many career questions come from students in India.
- **Missing data:** 24 comments had no text; they were dropped in cleaning.
- **Licensing and privacy:**
  - data comes through the official API under YouTube's terms;
  - author names are never stored, only a salted hash (HMAC), used to count different people.
