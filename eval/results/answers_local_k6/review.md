# Read-through of the k = 6 answers

**What this is.** A read-through of all 63 answers the local model gave to answerable questions in
`results.jsonl`, and of every citation that does not point to the gold-evidence passage, each
checked against the cited passages of the minutes. It was carried out by the author's AI
assistant (Claude) while building the project, not by a person, so it is a second opinion on the
automatic grades rather than ground truth. A human review of these notes is a next step.

**Why it was needed.** The automatic grader checks that required facts and a gold citation are
present. It cannot see a false statement added next to correct facts, so it can be too lenient.

## Answer-level findings

| Id | Automatic grade | Finding |
|---|---|---|
| sf10 | correct | **Factual error.** Says council rescinded the April 21, 2026 decision on May 5; the motion to rescind was lost 2 to 8, so the decision not to designate stood. The first and third sentences are right. |
| nl03 | correct | **Factual error.** Opens with "Seven (7) food trucks"; the minutes approve seven restaurants: five food trucks, one ice cream shop and one tent (the answer then lists these correctly). |
| vo06 | correct | Misdescribes the first motion (it was a motion to reject the parking relief, which lost 1 to 8); the final decision (relief approved 8 to 1) is right. |
| df02 | correct | Says the application was "initially rejected"; in fact the motion to reject failed 1 to 9 and the application was then approved 9 to 1 (stated correctly). |
| vo03 | correct | Minor: calls Deputy Mayor O'Leary "Councillor O'Leary". |
| pa06 | correct | Minor: presents the October 31, 2023 motion (to consider and advertise the amendment) as passing it; it also gives the actual adoption date, February 20, 2024. |
| sf04 | correct | Accurate but incomplete: does not mention that Amendment 23 was adopted again later the same day. |
| others | correct | No problems found in the remaining 56 answers. |

Counting the two factual errors as wrong, strict correctness would be 61 of 65 (93.8%) instead of
63 of 65; counting the two mischaracterised motions as well, 59 of 65 (90.8%).

## Citations outside the gold passage (16 of 82)

| Id | Cited passage | Verdict |
|---|---|---|
| sf04 | 2023-08-21, item 13.2 (notice of motion to rescind) | Supports the sentence about the notice |
| sf10 | 2026-05-05, item 13.2 (another part) | Accompanies the false "rescinded" sentence |
| sf10 | 2026-06-02, item 6.1.2 | Supports the June 2 sentence |
| sf11 | 2025-02-25, item 13.1 (discussion part) | Supports "in response to the trade war" |
| sf12 | 2024-09-17, item 8.1.4 | Supports the sentence on the 2024 appointments |
| vo02 | 2023-02-20, item 8.1.2 (part 2) | Supports "the majority of Council ... keep the current footprint" |
| vo02 | 2023-02-20, item 8.1.2 (part 1) | Background only |
| vo04 | 2023-06-19, item 6.4 (discussion part) | Background only |
| vo06 | 2023-10-03, item 6.1 (part 1) | Accompanies the misdescribed first motion |
| vo07 | 2026-08-11, item 6.1 | Supports the sentence on the 2026 warehouse approval |
| df02 | 2024-05-14, item 6.6 (another part) | Accompanies "initially rejected" |
| df09 | 2026-03-25, item 14.1 (motion part) | Supports "approved by Council on March 25, 2026" |
| pa01 | 2023-01-23, item 13.2 (notice of motion) | Background only |
| pa05 | 2023-12-12, item 13.1 | Supports the sentence on the December contract |
| pa06 | 2024-02-20, item 15.2 | Supports the February 20, 2024 adoption |
| pa08 | 2026-05-05, item 6.1 | Supports an accurate but off-topic sentence (150 Howlett's Line) |

With the 66 gold-evidence citations, 76 of 82 citations (92.7%) point to a passage that supports
a statement in the answer; 3 are background only and 3 accompany a misstatement. The automatic
measures are 80.5% (gold evidence) and 90.2% (gold agenda item).
