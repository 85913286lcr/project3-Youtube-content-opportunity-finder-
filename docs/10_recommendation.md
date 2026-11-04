# 10. Recommendation: Deploy or Not Deploy

Slide 11.

## Recommendation
**Deploy now as a decision aid. Roll out the ranking only if the A/B test wins.**

The creator uses the dashboard every week and makes the final pick. The ranking is not trusted to choose videos on its own until the A/B test shows it brings more views.

## Justification
| | Evidence |
|---|---|
| Performance | The filter beat the baseline on unseen October data (F0.5 0.71 vs 0.45). Topics are steadier by meaning and still fit 78–88% of new questions. Picks stayed in demand 7 in 10 times vs under 5 in 10 for random. |
| Risk | Low: a wrong pick costs one video slot, and a person approves every pick. |
| Opportunity size | Saves 2.5 h a week (130 h a year) and pays back in about 6 months. |

## Risks and limitations
- **Too few labels (main limit):**
  - only 356 training labels and 19 test questions;
  - next step: label 500 more.
- **Wrong predictions:**
  - false questions can make a topic look bigger;
  - missed questions (no "?") make it look smaller;
  - the creator reads the evidence before choosing.
- **Comments are not views:** high demand does not prove high views. This is why the ranking needs the A/B test.
- **Drift:**
  - only 78–88% of October questions still fit the topics, so new themes appear;
  - topic fit is monitored weekly and the topics are re-clustered monthly.
- **Bias:**
  - English only, 3 niches;
  - search results favour big channels;
  - career demand comes partly from students in India.
- **Regulatory and privacy:**
  - data is collected through the official API under YouTube's terms;
  - authors are stored only as a salted hash;
  - selling the tool would need a review of the API terms.
