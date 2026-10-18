# Evaluation

This folder holds the evaluation set, the relevance judgements used by CI, and the
recorded results. Numbers are in [`results/RESULTS.md`](results/RESULTS.md), which is
generated from the JSON files next to it.

## The question set (`questions.jsonl`)

77 questions about Regular Council Meeting minutes from January 2023 to September 2026:

| Type | Count | What it tests |
|---|---|---|
| `single_fact` | 14 | A named fact: who was appointed, which company won a contract, why an application was rejected |
| `number_lookup` | 13 | Amounts, parking spaces, hectares, budget totals, speed limits |
| `vote_outcome` | 14 | Carried or lost, the vote count, who voted against |
| `date_filtered` | 11 | The same question asked with different meeting-date ranges (budgets, the Tely 10, 319 Hamilton Avenue) |
| `paraphrase` | 13 | Wording that avoids the terms used in the minutes ("dig gravel on farmland" for a quarry referral) |
| `unanswerable` | 12 | Plausible topics that never appear in the minutes (rent control, a 2027 budget, ride-sharing) |

The 65 answerable questions have gold meetings in 2023 (29), 2024 (12), 2025 (12) and 2026 (12);
two of them have more than one gold item (a decision that was later rescinded or revisited).
Unanswerable questions are about 16% of the set.

Each answerable item records the gold meeting (date, eSCRIBE meeting id and, when one exists,
the minutes PDF document id), the agenda item reference, one or more **evidence** strings copied
verbatim from that item, a short reference answer, and **answer facts** (strings a correct answer
must contain, used by the answer grader). Each unanswerable item lists **absent terms** that do
not occur anywhere in the corpus.

**Provenance.** The questions were drafted with an AI assistant (Claude) by reading the parsed
minutes, then every gold reference was checked by code (below). They have not yet been reviewed
by a person; reviewing a stratified sample is a recommended next step. Reference answers were not
produced by the model being evaluated.

**Why agenda items, not pages.** The project brief asked for page-level gold references. The
City's PDF endpoint answers automated clients with a bot challenge (see the main README), so the
minutes are read from the official HTML rendering, which has no page numbers. The agenda item
is the unit of citation instead, and every item has a stable anchor on the official page.

## Verification

```bash
council-rag verify-eval
```

For every gold reference the command checks that the meeting exists with the stated date and
PDF id, that the agenda item exists, and that every evidence string occurs in that item's text
(case, accent and whitespace-insensitive). It checks that date-filtered gold meetings fall inside
the filter, that no absent term of an unanswerable question occurs in any meeting, and that
reference answers stay under 220 characters. The current set passes with no problems.

One item was adjusted after the first retrieval run: `vo06` originally required the evidence of
two motions that the chunker places in different chunks, which made it impossible to retrieve as
a whole. Its evidence was narrowed to the deciding motion (noted in the item). This moved hybrid
hit@5 from 0.939 to 0.954 (one question).

## Relevance and metrics

A retrieved chunk is **relevant** to a gold reference when it comes from the same meeting and
agenda item and contains every evidence string. This is the agenda-item counterpart of "the
correct source page" in NFR-01. An item-level variant (same meeting and item, any part) is
reported as `item_hit@5` in the JSON files.

* **hit@k**: share of answerable questions with at least one relevant chunk in the top k.
* **recall@5**: share of gold references with a relevant chunk in the top 5, averaged over questions.
* **MRR@10**: mean reciprocal rank of the first relevant chunk within the top 10.
* **95% CI**: Wilson score interval for hit@5. With 65 questions the interval is about plus or
  minus 7 points, so differences of one or two questions are not meaningful.
* **Latency**: wall-clock time per query on CPU, including query embedding for dense and hybrid.

`qrels.json` stores the relevant chunk ids per question for the default chunking. CI uses it
with the committed browser index, so it needs no City documents.

## Reproducing

From the repository root, with the virtual environment active:

```bash
council-rag download          # polite: about 10 minutes for ~100 meetings, cached afterwards
council-rag parse
council-rag build-index
council-rag verify-eval
council-rag eval-retrieval --configs main --write-qrels     # BM25, dense, hybrid (+ qrels)
council-rag eval-retrieval --configs ablations              # one setting changed per row
council-rag export-demo --out docs/data
council-rag eval-retrieval --source demo --tag demo_index_fp32_query
council-rag eval-retrieval --source demo --query-model onnx/model_quantized.onnx --tag demo_index_q8_query
npm install && python scripts/embedding_parity.py           # Python vs transformers.js vectors
python scripts/check_search_parity.py                       # Python vs browser ranking code
node scripts/screenshots.mjs                                # headless Chrome checks
python scripts/make_results_md.py                           # regenerate results/RESULTS.md
```

The CI gate (`.github/workflows/ci.yml`) runs
`council-rag eval-retrieval --source demo --gate 0.85`, which fails the build if hybrid hit@5 on
the committed index drops below the NFR-01 threshold or if the date filter returns a result
outside the requested range.

## Known limitations

* 65 answerable questions give wide confidence intervals; treat single-question differences as noise.
* The ablations reuse the development questions. Larger chunks (400 words) and RRF k=10 looked
  better here but were not adopted without a held-out check.
* Questions skew toward 2023 (29 of 65), the year read in most detail while drafting.
* The questions were AI-drafted and machine-verified, not yet human-reviewed.
