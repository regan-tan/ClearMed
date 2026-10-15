# Architecture Diagrams

ClearMed uses a Python 3.11 `src/` package for research code and scripts, with
generated datasets under `data/`. The React/TypeScript Vite frontend is an
independent preview client: `api/mock.ts` supplies MCQs and retrieval previews,
while `api/notebook.ts` owns the local notebook-preview adapter. Neither is a
backend integration layer.

The completed TF-IDF Step 1 path uses saved `part1_train` and `internal_dev`
partitions, leakage checks, and manifest verification before training. The
39.84% result is internal development only; future validation remains separate.

## Planned Diagrams

1. **Overall architecture**

   ```text
   MedMCQA → preprocessing → TF-IDF / Scratch Seq2Seq / FLAN-T5 + LoRA
     → QA evaluation → answer + explanation → student follow-up
     → MiniLM / E5 / BioLORD → top-k evidence → sufficient evidence?
       YES → FLAN-T5 + LoRA → grounded answer
       NO  → fixed fallback
   ```

2. **LoRA training workflow**

   ```text
   local development → small FLAN-T5 smoke test → school GPU cluster
     → LoRA fine-tuning → save adapter + predictions + metrics → local evaluation
   ```

3. **Retrieval pipeline**

   ```text
   context-rich follow-up query → embedding model → query vector
     → cosine similarity → rank MedMCQA training passages → top-k evidence
   ```

4. **Conversational retrieval and fallback flow**

   ```text
   retrieval → sufficient evidence?
     YES → FLAN-T5
     NO  → deterministic fallback
   ```
