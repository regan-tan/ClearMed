# Retrieval relevance labelling guideline

You are judging whether a retrieved passage would help a tutor answer a
student's follow-up question about an MCQ. Your labels decide which embedding
model retrieves best (RQ3) and set the threshold that stops the tutor from
using bad evidence. Both labellers must read and agree on this page before
starting.

## What each row shows

| Column | Meaning |
|---|---|
| `labeller` | Whose row this is. Filter the sheet to your own name and label only those rows. |
| `follow_up` | What the student asked after seeing the MCQ and its answer. **This is what you judge against.** |
| `question` | The MCQ with options A–D. |
| `correct_answer` | The correct option. |
| `reference_explanation` | The official explanation, to help you understand the question. It is a reference, not the target: a passage can be relevant without matching it. About half the evaluation questions have none in the dataset; then judge from the question, the answer and your own knowledge, and look it up if unsure. |
| `passage` | The retrieved evidence: another MCQ's question, answer and explanation. |
| `relevant` | Your label: `1` or `0`. |
| `note` | Optional. Write anything you were unsure about. |

Passages from the same MCQ appear together, in random order. You cannot see
which model found a passage. Do not try to guess, and do not look at the other
labeller's rows.

## The rule

Mark **1 (relevant)** if the passage contains at least one fact that a tutor
would actually use when answering *this follow-up* for *this question*.

Mark **0 (not relevant)** otherwise, including when:

- the passage is on the same topic but does not help with this follow-up;
- it is about a different drug, disease or structure, even a similar one;
- it would mislead the student (it contradicts the correct answer);
- the follow-up is off-topic ("What is the capital of France?"). **Always 0.**

Judge the information, not the passage's own MCQ. A passage whose question is
different can still be relevant if its explanation states the needed fact.
"Ans. (c)" at the start of a passage refers to the passage's own question;
ignore it.

## By follow-up type

| Follow-up | Relevant if the passage… |
|---|---|
| Why is that the correct answer? | explains or states why the correct option is right. |
| Why not option X? | explains why X is wrong, or what X actually is or does. If X *is* the correct answer, treat it as "why is X correct?". |
| What is the key clue in this question? | explains the finding or term in the question that points to the answer. |
| Can you explain the mechanism behind the answer? | describes how or why the correct answer works (pathway, cause, action). |
| Off-topic | never. Always 0. |

## Examples

MCQ: *Which drug should be avoided in a patient with asthma?* Correct answer: *Propranolol.*
Follow-up: *Can you explain the mechanism behind the answer?*

| Passage says | Label | Why |
|---|---|---|
| Non-selective beta blockers block β2 receptors in the bronchi and can cause bronchospasm. | 1 | States the mechanism. |
| Propranolol is used for migraine prophylaxis and essential tremor. | 0 | Right drug, but nothing on why it is dangerous in asthma. |
| Salbutamol is a β2 agonist used to relieve bronchospasm. | 1 | Explains the β2 receptor's role in the airway, which a tutor would use. |
| Atenolol is cardioselective and preferred in hypertension. | 0 | Different drug, and does not explain the mechanism asked about. |

When it is truly 50/50, choose 0 and write why in `note`.

## How to work

1. Upload `labelling_sheet.csv` to Google Sheets (File → Import → Upload).
2. Select the `relevant` column → Data → Data validation → Dropdown → `0`, `1`.
3. Data → Create a filter, then filter `labeller` to your name. Stage A rows
   come first; do them first. They are needed earlier, for the threshold.
4. Do not edit, delete, re-order or add rows, and do not change `item_id`,
   `labeller` or `stage`. Sorting or filtering is fine.
5. When both of you are done, File → Download → CSV, and save it as
   `data/retrieval_eval/labels/<pool_id>/labelling_sheet.csv`. Then run
   `uv run python scripts/import_labels.py --pool-id <pool_id>`.
6. The import checks every label. If you disagreed on any shared passage, it
   writes `disagreements.csv`. Discuss each one together, fill
   `agreed_label`, save it as `adjudication.csv` in the same folder and run the
   import again.

Take a break every 45 minutes. Labels get worse when you are tired.
