# RQ4 grading guide

You are grading replies from a medical-exam tutor. A student answered a multiple-choice question
(MCQ) and then asked a follow-up such as "Why is that the correct answer?". The tutor replied.
Your scores decide whether giving the tutor evidence makes its replies better.

You do **not** need to be a medical expert. Each row shows the correct answer and, when the
dataset has one, the textbook explanation. Judge the reply against those.

## The two sheets

| Sheet | Rows | What you score |
|---|---|---|
| `blind_sheet.csv` | every tutor reply, shuffled | `correct`, `no_unsupported_claims`, and `consistent` (only rows with a `previous_turn`) |
| `grounding_sheet.csv` | replies that were given evidence, with that `evidence` | `grounded` |

You never see which tutor version wrote a reply. That is on purpose.

## Rules for the spreadsheet

- Fill **only** the score columns (and `note` if you want). Use the digits `0`, `1` or `2`.
- Do **not** change, delete, reorder or add rows or columns. Never edit `item_id` or `labeller`.
- In the blind sheet, rows without a `previous_turn` say `n/a` under `consistent`. Leave them.
- Some replies appear twice, once for `labeller_1` and once for `labeller_2`. That is how we measure
  agreement. Grade each row on its own.
- Save as CSV (UTF-8) with the same file name.

## Scores

Read the whole reply, including the first line `Answer: X`.

**`correct`** — is the medicine in the reply right?
- **2**: right, and consistent with the correct answer shown in the row.
- **1**: partly right: a small error, or right facts that do not really answer the student, or the
  `Answer:` line and the explanation disagree.
- **0**: wrong, nonsense, or empty.

**`no_unsupported_claims`** — does the reply avoid making things up?
- **2**: every claim is true or supported by the correct answer and explanation.
- **1**: one doubtful or unsupported claim.
- **0**: clearly invents facts.

**`consistent`** (second turns only) — does the reply agree with the tutor's own earlier reply in
`previous_turn`?
- **2**: agrees. **1**: slightly contradicts it. **0**: contradicts it.

**`grounded`** (grounding sheet) — is the reply based on the `evidence` it was given?
- **2**: everything it says comes from the evidence.
- **1**: partly from the evidence.
- **0**: not based on the evidence (ignores it, or says things the evidence does not contain).

A reply can be grounded but wrong (the evidence was about something else) or correct but not
grounded (true facts the evidence does not mention). Score each column on its own.

## Two worked examples

**Example A.** MCQ: "Dilated pupil with coma is seen in which poisoning? A. Opium B. Datura
C. Mushroom D. Pilocarpus". Correct answer: B. Datura. Follow-up: "Why not option A?"
Reply: "Answer: A Explanation: Opium poisoning is characterized by dilated pupil with facial flush."
Evidence [1] says opium causes miosis (small pupils) and datura causes dilated pupils.
- `correct` **0**: opium causes small pupils, not dilated ones.
- `no_unsupported_claims` **0**: the claim is invented.
- `grounded` **0**: the evidence says the opposite.

**Example B.** MCQ: "Thorium induced tumor? A. Angiosarcoma of liver …". Correct answer: A.
Follow-up: "Why not option D?" Reply: "Answer: A Explanation: Hepatic angiosarcomas are associated
with carcinogen exposures, including arsenic, thorotrast and polyvinyl chloride."
Evidence says thorotrast (thorium dioxide) causes liver angiosarcoma.
- `correct` **1**: true and supports A, but never says why D (astrocytoma) is wrong.
- `no_unsupported_claims` **2**.
- `grounded` **2**.

## Grading with LLMs

If you use LLMs, use **two different LLMs**: one for every `labeller_1` row and another for every
`labeller_2` row. Then check about 20 rows by hand and fix what is wrong. The report will say the
grades are LLM-generated with a human spot-check.

Paste the prompt below, then the rows (a few at a time works best). Ask for the scores back in
the same row order and copy them into the sheet yourself.

```text
You are grading replies from a medical-exam tutor. For each row I give you: the MCQ, the correct
answer, a reference explanation (may be missing), the student's follow-up, the tutor's previous
reply (second turns only), the evidence the tutor was given (grounding rows only) and the reply.

Score with integers only:
- correct: 2 = medically right and consistent with the correct answer; 1 = partly right, small
  error, does not really answer the follow-up, or its "Answer:" line contradicts its explanation;
  0 = wrong, nonsense or empty.
- no_unsupported_claims: 2 = no invented or unsupported claims; 1 = one doubtful claim; 0 = clearly
  invents facts.
- consistent (only if a previous reply is given): 2 = agrees with it; 1 = slightly contradicts;
  0 = contradicts.
- grounded (only if evidence is given): 2 = everything comes from the evidence; 1 = partly;
  0 = not based on the evidence.

Judge only against the correct answer, the reference explanation and the evidence. Do not reward
fluent writing. Return one line per row: item_id, then the scores in the order listed above, using
"n/a" where a score does not apply. No explanations.
```

## When you are done

Save the completed sheets as:

```
data/retrieval_eval/rq4/<run_id>/blind_sheet.csv
data/retrieval_eval/rq4/<run_id>/grounding_sheet.csv
```

Then run `uv run python scripts/evaluate_conversation.py --run-id <run_id>`. It refuses sheets
with missing or invalid scores and tells you which rows to fix.
